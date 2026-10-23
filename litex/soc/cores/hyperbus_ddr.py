#
# This file is part of LiteX.
#
# Copyright (c) 2026 Florent Kermarrec <florent@enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""HyperRAM DDR controller (AXI4, streaming oriented).

Unlike `litex.soc.cores.hyperbus.HyperRAM` (fabric oversampling PHY, CK = PHY clock / 4), this
controller uses DDR IO registers: the core runs at the HyperRAM clock frequency and transfers one
full DDR word (2 x DQ width) per cycle, so a x16 HyperRAM at 200MHz gives 800MB/s peak (~720MB/s
writes / ~700MB/s reads measured with 1KB bursts on Efinix Ti60 F225 Dev Kit).

Architecture:
- AXI4 slave (INCR bursts, data width = 2 x DQ width), AW/AR command queues, bursts split to
  respect tCSM, read/write arbitration, W/R data FIFOs: a command is only issued when its whole
  write data is buffered (or when its whole read data can be stored), so CK never stalls.
- Core FSM: CA (3 CK) -> fixed 2x latency -> data -> CS high / tRWR.
- PHY interface (`HyperRAMDDRPHYInterface`): per core cycle, CS#/CK enable/DQ/RWDS output words
  (HI = first edge, LO = second edge) and DDR captured DQ/RWDS input words (HI = first in time) in
  a separate capture clock domain. Read words are extracted from the RWDS strobe pattern (both
  word alignments supported) and cross to the core domain through an async FIFO.
- Read calibration (`HyperRAMDDRCalibration`): when the PHY can shift its capture clock phase, a
  pattern is written/read back at startup for each phase and the center of the passing window is
  selected (gateware only, no software needed).
- `HyperRAMTitaniumDDRPHY`: Efinix Titanium GPIO DDIO implementation (capture phase through the
  PLL dynamic phase shift).
"""

import math

from migen import *
from migen.genlib.cdc import MultiReg

from litex.gen import *

from litex.soc.interconnect import axi, stream
from litex.soc.interconnect.csr import *

# Constants ----------------------------------------------------------------------------------------

HYPERRAM_DDR_TRWR   = 40e-9  # Read-Write recovery time (s).
HYPERRAM_DDR_TCSM   = 4e-6   # Max CS# low time (s).
HYPERRAM_DDR_TVCS   = 150e-6 # Power-up/reset to first access (s).

# PHY Interface ------------------------------------------------------------------------------------

class HyperRAMDDRPHYInterface:
    """Signals between HyperRAMDDRCore and a DDR PHY.

    Core clock domain (one cycle = one CK period):
    - cs_n, rst_n, ck_en         : CK toggles (rising then falling edge) during cycles with ck_en.
    - dq_o_hi/lo, dq_oe          : DQ for first (rising CK) / second (falling CK) edge.
    - rwds_o_hi/lo, rwds_oe      : RWDS (write mask) for first / second edge.
    Capture clock domain (`cd_cap`):
    - dq_i_hi/lo, rwds_i_hi/lo   : DDR captured inputs (HI = first half-word in time).
    """
    def __init__(self, dq_width=16, cd_cap="hyperram_cap"):
        self.dq_width = dq_width
        self.cd_cap   = cd_cap
        rwds_width    = dq_width//8
        self.cs_n      = Signal(reset=1)
        self.rst_n     = Signal(reset=1)
        self.ck_en     = Signal()
        self.dq_o_hi   = Signal(dq_width)
        self.dq_o_lo   = Signal(dq_width)
        self.dq_oe     = Signal()
        self.rwds_o_hi = Signal(rwds_width)
        self.rwds_o_lo = Signal(rwds_width)
        self.rwds_oe   = Signal()
        self.dq_i_hi   = Signal(dq_width)
        self.dq_i_lo   = Signal(dq_width)
        self.rwds_i_hi = Signal(rwds_width)
        self.rwds_i_lo = Signal(rwds_width)

# Read Capture -------------------------------------------------------------------------------------

class _HyperRAMDDRCapture(LiteXModule):
    """Extract read words from DDR captured DQ/RWDS (runs in the capture clock domain).

    During a read, the device drives RWDS high with the first half-word and low with the second
    one. Depending on the capture clock phase, a word is either fully captured in one cycle
    (HI=first, LO=second) or straddles two cycles (LO of cycle n = first, HI of cycle n+1 =
    second): both alignments are supported.
    """
    def __init__(self, phy_if):
        dw = phy_if.dq_width
        self.enable = Signal()
        self.source = source = stream.Endpoint([("data", 2*dw)])

        # # #

        rwds_hi = phy_if.rwds_i_hi[0]
        rwds_lo = phy_if.rwds_i_lo[0]

        # Register inputs.
        dq_hi   = Signal(dw)
        dq_lo   = Signal(dw)
        s_hi    = Signal()
        s_lo    = Signal()
        en      = Signal()
        self.sync += [
            dq_hi.eq(phy_if.dq_i_hi),
            dq_lo.eq(phy_if.dq_i_lo),
            s_hi.eq(rwds_hi),
            s_lo.eq(rwds_lo),
            en.eq(self.enable),
        ]

        # Odd alignment: first half in LO, second half in next HI.
        pending      = Signal()
        pending_data = Signal(dw)
        self.sync += [
            source.valid.eq(0),
            pending.eq(0),
            If(en,
                If(s_hi & ~s_lo,
                    # Even alignment.
                    source.valid.eq(1),
                    source.data.eq(Cat(dq_lo, dq_hi)),
                ).Elif(pending & ~s_hi,
                    source.valid.eq(1),
                    source.data.eq(Cat(dq_hi, pending_data)),
                ),
                If(s_lo & ~s_hi,
                    pending.eq(1),
                    pending_data.eq(dq_lo),
                ),
            ),
        ]

# AXI Burst Splitter -------------------------------------------------------------------------------

class _HyperRAMDDRSplitter(LiteXModule):
    """Split AXI AW/AR INCR bursts in HyperRAM commands of at most `max_words` words."""
    def __init__(self, ax, max_words, word_bytes, id_width):
        self.source = source = stream.Endpoint([
            ("addr", 32), ("len", 9), ("id", id_width)])

        # # #

        addr   = Signal(32)
        left   = Signal(9) # Remaining words (1-256).
        txn_id = Signal(id_width)
        busy   = Signal()
        n      = Signal(9)

        self.comb += [
            n.eq(Mux(left > max_words, max_words, left)),
            source.valid.eq(busy),
            source.addr.eq(addr),
            source.len.eq(n),
            source.id.eq(txn_id),
            source.last.eq(left == n),
            ax.ready.eq(~busy),
        ]
        self.sync += [
            If(ax.valid & ax.ready,
                busy.eq(1),
                addr.eq(ax.addr),
                left.eq(ax.len + 1),
                txn_id.eq(ax.id),
            ),
            If(source.valid & source.ready,
                addr.eq(addr + n*word_bytes),
                left.eq(left - n),
                If(left == n, busy.eq(0)),
            ),
        ]

# HyperRAM DDR Core --------------------------------------------------------------------------------

class HyperRAMDDRCore(LiteXModule):
    """HyperRAM DDR controller core (AXI4 slave).

    Parameters:
    - phy_if       : HyperRAMDDRPHYInterface.
    - clk_freq     : Core (= HyperRAM CK) frequency.
    - latency      : HyperRAM initial latency (CK cycles, fixed 2x latency used).
    - axi_id_width : AXI ID width.
    - fifo_depth   : W/R data FIFOs depth (words), must be >= max burst words.
    - max_words    : Max words per HyperRAM burst (default: from tCSM, at most 256).
    - size         : HyperRAM size in bytes (AXI addresses are masked to it, ex to ignore the SoC
                     region base).
    - with_continuation : Continue the HyperRAM burst with the next AXI command when sequential and
                     in the same direction (CS# kept low up to tCSM, saving CA/latency/tRWR).
    - with_reset   : Pulse RST# and wait tVCS at startup.
    """
    def __init__(self, phy_if, clk_freq, latency=7, axi_id_width=4, fifo_depth=512, max_words=None,
        size=None, with_continuation=True, with_reset=True, with_csr=True):
        dw         = phy_if.dq_width
        word_width = 2*dw
        word_bytes = word_width//8
        if max_words is None:
            # Largest power of two burst keeping CS# low time (CA + latency + data) below tCSM.
            tcsm_words = int(HYPERRAM_DDR_TCSM*clk_freq) - 3 - 2*latency - 8
            max_words  = 256
            while max_words > tcsm_words and max_words > 1:
                max_words //= 2
        if max_words < 1 or max_words > 256:
            raise ValueError("HyperRAM clock too low for tCSM.")
        if fifo_depth < max_words:
            raise ValueError("HyperRAM DDR fifo_depth must be >= max_words.")
        # Max words per CS# low period (CA + latency + data below tCSM).
        cs_words_max = max(max_words, int(HYPERRAM_DDR_TCSM*clk_freq) - 3 - 2*latency - 8)
        # CS# high time between accesses: tRWR, and at least the read capture window close delay
        # (+2: CS# hold cycles added by the PHY after the last CK cycle).
        trwr_cycles = max(4, math.ceil(HYPERRAM_DDR_TRWR*clk_freq)) + 2

        self.bus = bus = axi.AXIInterface(
            data_width    = word_width,
            address_width = 32,
            id_width      = axi_id_width,
        )
        self.max_words = max_words

        # Config / Status.
        self.latency      = Signal(4, reset=latency)
        self.rd_timeouts  = Signal(32)
        self.ready        = Signal()

        # Register interface (single access).
        self.reg_start = Signal()
        self.reg_we    = Signal()
        self.reg_addr  = Signal(2)
        self.reg_wdata = Signal(16)
        self.reg_rdata = Signal(16)
        self.reg_done  = Signal()

        # # #

        # Startup: RST# pulse + tVCS.
        # ---------------------------
        if with_reset:
            startup_cycles = int((HYPERRAM_DDR_TVCS + 2e-6)*clk_freq)
            startup = Signal(max=startup_cycles + 1, reset=startup_cycles)
            self.sync += If(startup != 0, startup.eq(startup - 1))
            self.comb += [
                phy_if.rst_n.eq(startup < (startup_cycles - int(1e-6*clk_freq))),
                self.ready.eq(startup == 0),
            ]
        else:
            self.comb += self.ready.eq(1)

        # Command splitters.
        # ------------------
        self.aw_split = aw_split = _HyperRAMDDRSplitter(bus.aw, max_words, word_bytes, axi_id_width)
        self.ar_split = ar_split = _HyperRAMDDRSplitter(bus.ar, max_words, word_bytes, axi_id_width)

        # Write data FIFO.
        # ----------------
        self.w_fifo = w_fifo = stream.SyncFIFO([("data", word_width), ("strb", word_bytes)], fifo_depth, buffered=True)
        self.comb += [
            w_fifo.sink.valid.eq(bus.w.valid),
            bus.w.ready.eq(w_fifo.sink.ready),
            w_fifo.sink.data.eq(bus.w.data),
            w_fifo.sink.strb.eq(bus.w.strb),
        ]

        # B responses.
        # ------------
        self.b_fifo = b_fifo = stream.SyncFIFO([("id", axi_id_width)], 8)
        self.comb += [
            bus.b.valid.eq(b_fifo.source.valid),
            bus.b.id.eq(b_fifo.source.id),
            bus.b.resp.eq(axi.RESP_OKAY),
            b_fifo.source.ready.eq(bus.b.ready),
        ]

        # Read data: capture (cd_cap) -> async FIFO -> R FIFO.
        # ----------------------------------------------------
        self.capture = capture = ClockDomainsRenamer(phy_if.cd_cap)(_HyperRAMDDRCapture(phy_if))
        cap_en = Signal()
        self.specials += MultiReg(cap_en, capture.enable, odomain=phy_if.cd_cap, n=2)
        if phy_if.cd_cap == "sys":
            cap_source = capture.source
        else:
            self.cap_cdc = cap_cdc = stream.ClockDomainCrossing([("data", word_width)],
                cd_from = phy_if.cd_cap,
                cd_to   = "sys",
                depth   = 16,
            )
            self.comb += capture.source.connect(cap_cdc.sink)
            cap_source = cap_cdc.source

        self.r_fifo = r_fifo = stream.SyncFIFO([("data", word_width), ("resp", 2)], fifo_depth, buffered=True)
        rd_cmd = stream.SyncFIFO([("len", 9), ("id", axi_id_width)], 4)
        self.rd_cmd = rd_cmd

        # Words routed to the register interface or R FIFO; timeouts inject error beats.
        rd_reg        = Signal()          # Current read is a register read.
        rd_expected   = Signal(11)        # Words still expected for the current read.
        rd_add        = Signal(9)         # Words added to the current read (continuation).
        rd_inject     = Signal()
        self.comb += [
            r_fifo.sink.valid.eq(~rd_reg & (((cap_source.valid & (rd_expected != 0)) | rd_inject))),
            r_fifo.sink.data.eq(cap_source.data),
            r_fifo.sink.resp.eq(Mux(rd_inject, axi.RESP_SLVERR, axi.RESP_OKAY)),
            cap_source.ready.eq(1),
        ]

        # R channel: beats from R FIFO, id/last from read commands.
        r_count = Signal(9)
        self.comb += [
            bus.r.valid.eq(r_fifo.source.valid & rd_cmd.source.valid),
            bus.r.data.eq(r_fifo.source.data),
            bus.r.resp.eq(r_fifo.source.resp),
            bus.r.id.eq(rd_cmd.source.id),
            bus.r.last.eq((r_count == (rd_cmd.source.len - 1)) & rd_cmd.source.last),
            r_fifo.source.ready.eq(bus.r.ready & rd_cmd.source.valid),
            rd_cmd.source.ready.eq(bus.r.valid & bus.r.ready & (r_count == (rd_cmd.source.len - 1))),
        ]
        self.sync += If(bus.r.valid & bus.r.ready,
            r_count.eq(r_count + 1),
            If(r_count == (rd_cmd.source.len - 1), r_count.eq(0)),
        )

        # R FIFO space reservation (words reserved for issued reads, released when sent on R).
        r_reserved = Signal(max=fifo_depth + 257)
        r_reserve  = Signal(9)
        self.sync += r_reserved.eq(r_reserved + r_reserve - (bus.r.valid & bus.r.ready))

        # Write data level (words available in W FIFO).
        w_level = Signal(max=fifo_depth + 2)
        self.comb += w_level.eq(w_fifo.level)

        # Command Arbitration.
        # --------------------
        cmd_we     = Signal()
        cmd_reg    = Signal()
        cmd_addr   = Signal(32)
        cmd_len    = Signal(9)
        cmd_id     = Signal(axi_id_width)
        cmd_last   = Signal()
        last_write = Signal()

        w_ok = aw_split.source.valid & (w_level >= aw_split.source.len) & b_fifo.sink.ready
        r_ok = ar_split.source.valid & ((r_reserved + ar_split.source.len) <= fifo_depth) & rd_cmd.sink.ready
        sel_w = Signal()
        self.comb += sel_w.eq(Mux(w_ok & r_ok, ~last_write, w_ok))

        # Burst continuation: next command sequential, same direction, fits in tCSM.
        cs_words = Signal(max=cs_words_max + 257)
        next_w   = Signal()
        next_r   = Signal()
        if with_continuation:
            self.comb += [
                next_w.eq(
                    aw_split.source.valid &
                    (aw_split.source.addr == (cmd_addr + cmd_len*word_bytes)) &
                    (w_level >= (aw_split.source.len + 1)) &   # +1: current word still in FIFO.
                    (b_fifo.level < (b_fifo.depth - 1)) &
                    ((cs_words + aw_split.source.len) <= cs_words_max)
                ),
                next_r.eq(
                    ar_split.source.valid &
                    (ar_split.source.addr == (cmd_addr + cmd_len*word_bytes)) &
                    ((r_reserved + ar_split.source.len) <= fifo_depth) &
                    rd_cmd.sink.ready &
                    ((cs_words + ar_split.source.len) <= cs_words_max)
                ),
            ]

        # CA generation.
        # --------------
        ca      = Signal(48)
        waddr   = Signal(32)
        addr_mask = (2**32 - 1) if size is None else (size - 1)
        self.comb += waddr.eq((cmd_addr & addr_mask)[log2_int(word_bytes):])
        self.comb += If(cmd_reg,
            ca[47].eq(~cmd_we),
            ca[46].eq(1),
            ca[45].eq(1),
            Case(self.reg_addr, {
                0 : ca[0:40].eq(0x00_00_00_00_00),
                1 : ca[0:40].eq(0x00_00_00_00_01),
                2 : ca[0:40].eq(0x00_01_00_00_00),
                3 : ca[0:40].eq(0x00_01_00_00_01),
            }),
        ).Else(
            ca[47].eq(~cmd_we),
            ca[46].eq(0),
            ca[45].eq(1),             # Linear burst.
            ca[16:45].eq(waddr[3:]),  # Row & upper column address.
            ca[0:3].eq(waddr[0:3]),   # Lower column address.
        )
        ca_bytes = [ca[48 - 8*(i + 1):48 - 8*i] for i in range(6)]

        # FSM.
        # ----
        count     = Signal(10)
        data_wait = Signal(5)
        # CK cycles from end of CA to first data (fixed 2x latency).
        self.comb += data_wait.eq(2*self.latency - 1)

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(self.ready,
                If(self.reg_start,
                    NextValue(cmd_reg, 1),
                    NextValue(cmd_we,  self.reg_we),
                    NextValue(cmd_len, 1),
                    NextState("CA0"),
                ).Elif(w_ok | r_ok,
                    NextValue(cs_words, Mux(sel_w, aw_split.source.len, ar_split.source.len)),
                    NextValue(cmd_reg,    0),
                    NextValue(cmd_we,     sel_w),
                    NextValue(last_write, sel_w),
                    If(sel_w,
                        aw_split.source.ready.eq(1),
                        NextValue(cmd_addr, aw_split.source.addr),
                        NextValue(cmd_len,  aw_split.source.len),
                        NextValue(cmd_id,   aw_split.source.id),
                        NextValue(cmd_last, aw_split.source.last),
                    ).Else(
                        ar_split.source.ready.eq(1),
                        rd_cmd.sink.valid.eq(1),
                        rd_cmd.sink.len.eq(ar_split.source.len),
                        rd_cmd.sink.id.eq(ar_split.source.id),
                        rd_cmd.sink.last.eq(ar_split.source.last),
                        r_reserve.eq(ar_split.source.len),
                        NextValue(cmd_addr, ar_split.source.addr),
                        NextValue(cmd_len,  ar_split.source.len),
                    ),
                    NextState("CA0"),
                )
            )
        )
        for i in range(3):
            fsm.act(f"CA{i}",
                phy_if.cs_n.eq(0),
                phy_if.ck_en.eq(1),
                phy_if.dq_oe.eq(1),
                phy_if.dq_o_hi.eq(ca_bytes[2*i + 0]),
                phy_if.dq_o_lo.eq(ca_bytes[2*i + 1]),
                NextValue(count, 0),
                If(cmd_reg & cmd_we,
                    NextState("REG-WRITE")
                ).Else(
                    NextState("LATENCY")
                ) if i == 2 else NextState(f"CA{i+1}")
            )
        fsm.act("LATENCY",
            phy_if.cs_n.eq(0),
            phy_if.ck_en.eq(1),
            # Open read capture window ahead of the first data (RWDS is low during latency).
            cap_en.eq(~cmd_we & (count + 6 >= data_wait)),
            NextValue(count, count + 1),
            If(count == (data_wait - 1),
                NextValue(count, 0),
                If(cmd_we,
                    NextState("WRITE")
                ).Else(
                    NextValue(rd_reg, cmd_reg),
                    rd_add.eq(cmd_len),
                    NextState("READ")
                )
            )
        )
        fsm.act("REG-WRITE",
            phy_if.cs_n.eq(0),
            phy_if.ck_en.eq(1),
            phy_if.dq_oe.eq(1),
            phy_if.dq_o_hi.eq(self.reg_wdata[8:16]),
            phy_if.dq_o_lo.eq(self.reg_wdata[0:8]),
            self.reg_done.eq(1),
            NextState("END"),
        )
        fsm.act("WRITE",
            phy_if.cs_n.eq(0),
            phy_if.ck_en.eq(1),
            phy_if.dq_oe.eq(1),
            phy_if.rwds_oe.eq(1),
            phy_if.dq_o_hi.eq(w_fifo.source.data[dw:]),
            phy_if.dq_o_lo.eq(w_fifo.source.data[:dw]),
            phy_if.rwds_o_hi.eq(~w_fifo.source.strb[dw//8:]),
            phy_if.rwds_o_lo.eq(~w_fifo.source.strb[:dw//8]),
            w_fifo.source.ready.eq(1),
            NextValue(count, count + 1),
            If(count == (cmd_len - 1),
                If(cmd_last,
                    b_fifo.sink.valid.eq(1),
                    b_fifo.sink.id.eq(cmd_id),
                ),
                NextValue(count, 0),
                If(next_w,
                    aw_split.source.ready.eq(1),
                    NextValue(cmd_addr, aw_split.source.addr),
                    NextValue(cmd_len,  aw_split.source.len),
                    NextValue(cmd_id,   aw_split.source.id),
                    NextValue(cmd_last, aw_split.source.last),
                    NextValue(cs_words, cs_words + aw_split.source.len),
                ).Else(
                    NextState("END"),
                )
            )
        )
        fsm.act("READ",
            phy_if.cs_n.eq(0),
            phy_if.ck_en.eq(1),
            cap_en.eq(1),
            NextValue(count, count + 1),
            If(count == (cmd_len - 1),
                NextValue(count, 0),
                If(next_r & ~cmd_reg,
                    ar_split.source.ready.eq(1),
                    rd_cmd.sink.valid.eq(1),
                    rd_cmd.sink.len.eq(ar_split.source.len),
                    rd_cmd.sink.id.eq(ar_split.source.id),
                    rd_cmd.sink.last.eq(ar_split.source.last),
                    r_reserve.eq(ar_split.source.len),
                    rd_add.eq(ar_split.source.len),
                    NextValue(cmd_addr, ar_split.source.addr),
                    NextValue(cmd_len,  ar_split.source.len),
                    NextValue(cs_words, cs_words + ar_split.source.len),
                ).Else(
                    NextState("READ-WAIT"),
                )
            )
        )
        # Wait for the read words to cross the capture path (CS# high, no CK).
        read_timeout = 64
        fsm.act("READ-WAIT",
            cap_en.eq(1),
            NextValue(count, count + 1),
            If(rd_expected == 0,
                NextValue(count, 0),
                NextState("END"),
            ).Elif(count == (read_timeout - 1),
                NextValue(self.rd_timeouts, self.rd_timeouts + 1),
                NextValue(count, 0),
                NextState("READ-FLUSH"),
            )
        )
        # Timeout: complete the read with error beats so the AXI master never hangs.
        fsm.act("READ-FLUSH",
            If(rd_expected == 0,
                NextState("END"),
            ).Elif(rd_reg,
                self.reg_done.eq(1),
                rd_inject.eq(1), # Consumes the missing register word (not sent to the R FIFO).
            ).Else(
                rd_inject.eq(~cap_source.valid),
            )
        )
        fsm.act("END",
            NextValue(count, count + 1),
            If(count >= (trwr_cycles - 1),
                NextValue(count, 0),
                NextState("IDLE"),
            )
        )

        # Read words accounting.
        rd_dec = Signal()
        self.comb += rd_dec.eq((cap_source.valid | rd_inject) & (rd_expected != 0))
        self.sync += [
            rd_expected.eq(rd_expected - rd_dec + rd_add),
            If(rd_dec,
                If(rd_reg & cap_source.valid,
                    # Register data: MSB on first edge, LSB on second edge (DQ[7:0]).
                    self.reg_rdata.eq(Cat(cap_source.data[0:8], cap_source.data[dw:dw+8])),
                ),
            ),
        ]
        self.comb += If(fsm.ongoing("READ-WAIT") & rd_reg & (rd_expected == 0), self.reg_done.eq(1))

        # CSRs.
        # -----
        if with_csr:
            self.add_csr()

    def add_csr(self):
        self._config = CSRStorage(fields=[
            CSRField("latency", size=4, offset=0, reset=self.latency.reset.value, description="HyperRAM initial latency (CK)."),
        ])
        self._status = CSRStatus(fields=[
            CSRField("ready",     size=1, offset=0, description="Startup done."),
            CSRField("max_words", size=9, offset=8, reset=self.max_words, description="Max words per burst."),
        ])
        self._rd_timeouts = CSRStatus(32, description="Read timeouts (missing read words).")
        self._reg_control = CSRStorage(fields=[
            CSRField("write", size=1, offset=0, pulse=True, description="Issue register write."),
            CSRField("read",  size=1, offset=1, pulse=True, description="Issue register read."),
            CSRField("addr",  size=2, offset=8, description="0: ID0, 1: ID1, 2: CR0, 3: CR1."),
        ])
        self._reg_status = CSRStatus(fields=[CSRField("done", size=1, offset=0)])
        self._reg_wdata  = CSRStorage(16)
        self._reg_rdata  = CSRStatus(16)

        reg_pending = Signal()
        reg_is_write = Signal()
        self.comb += [
            self.latency.eq(self._config.fields.latency),
            self._status.fields.ready.eq(self.ready),
            self._rd_timeouts.status.eq(self.rd_timeouts),
            self._reg_rdata.status.eq(self.reg_rdata),
            self._reg_status.fields.done.eq(~reg_pending),
            self.reg_we.eq(reg_is_write),
            self.reg_addr.eq(self._reg_control.fields.addr),
            self.reg_wdata.eq(self._reg_wdata.storage),
            self.reg_start.eq(reg_pending & self.fsm.ongoing("IDLE")),
        ]
        self.sync += [
            If(self._reg_control.fields.write | self._reg_control.fields.read,
                reg_pending.eq(1),
                reg_is_write.eq(self._reg_control.fields.write),
            ).Elif(self.reg_done,
                reg_pending.eq(0),
            ),
        ]

# Efinix Titanium DDR PHY --------------------------------------------------------------------------

class HyperRAMTitaniumDDRPHY(LiteXModule):
    """HyperRAM DDR PHY on Efinix Titanium GPIOs (DDIO).

    Clocks (all at the HyperRAM CK frequency, from the same PLL):
    - `cd_sys`  : Core clock. DQ/RWDS/OE are registered on its falling edge (DDIO), CS#/RST# on its
                  rising edge.
    - `cd_90`   : 90° shifted clock. CK is generated by a DDIO register on its falling edge, so CK
                  edges are centered in the DQ/RWDS output data eyes.
    - `cd_cap`  : Read capture clock (DDIO resync inputs). Its phase sets the read capture point; it
                  can be adjusted at runtime through the PLL dynamic phase shift (`shift_pads`) to
                  calibrate reads.
    """
    def __init__(self, platform, pads, cd_sys="sys", cd_90="sys90", cd_cap="sys_cap", shift_pads=None,
        shift_sel=None, drive_strength=8):
        from litex.build.efinix.common import gpio_info, add_gpio_block

        dw = len(pads.dq)
        self.phy_if = phy_if = HyperRAMDDRPHYInterface(dq_width=dw, cd_cap=cd_cap)

        # # #

        def add_output(pad, clk, clk_inv, ddio):
            io_name, io_pad, io_prop = gpio_info(platform, pad)
            add_gpio_block(platform, {
                "type"           : "GPIO",
                "mode"           : "OUTPUT",
                "name"           : io_name,
                "location"       : io_pad,
                "properties"     : io_prop,
                "size"           : len(pad),
                "out_reg"        : "DDIO" if ddio else "REG",
                "out_clk_pin"    : clk,
                "out_clk_inv"    : int(clk_inv),
                "drive_strength" : dict(io_prop).get("DRIVE_STRENGTH", str(drive_strength)),
            }, pad)
            if ddio:
                return (platform.add_iface_io(io_name + "_HI", len(pad)),
                        platform.add_iface_io(io_name + "_LO", len(pad)))
            return platform.add_iface_io(io_name, len(pad))

        def add_inout(pad):
            io_name, io_pad, io_prop = gpio_info(platform, pad)
            add_gpio_block(platform, {
                "type"           : "GPIO",
                "mode"           : "INOUT",
                "name"           : io_name,
                "location"       : io_pad,
                "properties"     : list(io_prop) + [("PULL_OPTION", "WEAK_PULLDOWN")],
                "size"           : len(pad),
                "in_reg"         : "DDIO_RESYNC",
                "in_clk_pin"     : ClockSignal(cd_cap),
                "in_clk_inv"     : 0,
                "out_reg"        : "DDIO",
                "out_clk_pin"    : ClockSignal(cd_sys),
                "out_clk_inv"    : 1,
                "oe_reg"         : "REG",
                "drive_strength" : dict(io_prop).get("DRIVE_STRENGTH", str(drive_strength)),
            }, pad)
            return {s: platform.add_iface_io(f"{io_name}_{s}", len(pad))
                for s in ["OUT_HI", "OUT_LO", "OE", "IN_HI", "IN_LO"]}

        # Output timing: CS#/RST# use rising-edge registers (pin updated at the start of the next
        # cycle) while DQ/RWDS/CK DDIO registers sample on falling edges within the cycle: delay
        # DQ/RWDS/CK by one cycle so CS# falls 3/4 of a CK period before the first CK rising edge,
        # and keep CS# low `cs_hold` cycles after the last CK cycle (tCSH and last read word).
        sd = getattr(self.sync, cd_sys)
        ck_en     = Signal()
        dq_o_hi   = Signal(dw)
        dq_o_lo   = Signal(dw)
        dq_oe     = Signal()
        rwds_o_hi = Signal(dw//8)
        rwds_o_lo = Signal(dw//8)
        rwds_oe   = Signal()
        sd += [
            ck_en.eq(phy_if.ck_en),
            dq_o_hi.eq(phy_if.dq_o_hi),
            dq_o_lo.eq(phy_if.dq_o_lo),
            dq_oe.eq(phy_if.dq_oe),
            rwds_o_hi.eq(phy_if.rwds_o_hi),
            rwds_o_lo.eq(phy_if.rwds_o_lo),
            rwds_oe.eq(phy_if.rwds_oe),
        ]
        cs_hold = 2
        cs_n_sr = Signal(cs_hold, reset=2**cs_hold - 1)
        sd += cs_n_sr.eq(Cat(phy_if.cs_n, cs_n_sr))

        # CK (differential or single-ended).
        ck_pads = [pads.clk_p, pads.clk_n] if hasattr(pads, "clk_p") else [pads.clk]
        for i, ck_pad in enumerate(ck_pads):
            ck_hi, ck_lo = add_output(ck_pad, ClockSignal(cd_90), clk_inv=True, ddio=True)
            self.comb += [
                ck_hi.eq(ck_en if i == 0 else ~ck_en),
                ck_lo.eq(0     if i == 0 else 1),
            ]

        # CS#/RST#.
        cs_n  = add_output(pads.cs_n,  ClockSignal(cd_sys), clk_inv=False, ddio=False)
        rst_n = add_output(pads.rst_n, ClockSignal(cd_sys), clk_inv=False, ddio=False)
        self.comb += [
            cs_n.eq(phy_if.cs_n & (cs_n_sr == (2**cs_hold - 1))),
            rst_n.eq(phy_if.rst_n),
        ]

        # DQ/RWDS.
        dq   = add_inout(pads.dq)
        rwds = add_inout(pads.rwds)
        self.comb += [
            dq["OUT_HI"].eq(dq_o_hi),
            dq["OUT_LO"].eq(dq_o_lo),
            dq["OE"].eq(Replicate(dq_oe, dw)),
            rwds["OUT_HI"].eq(rwds_o_hi),
            rwds["OUT_LO"].eq(rwds_o_lo),
            rwds["OE"].eq(Replicate(rwds_oe, dw//8)),
            # DDIO_RESYNC inputs: IN_LO (falling edge) is captured half a cycle before IN_HI
            # (rising edge) of the same cycle, so IN_LO is the first half-word in time.
            phy_if.dq_i_hi.eq(dq["IN_LO"]),
            phy_if.dq_i_lo.eq(dq["IN_HI"]),
            phy_if.rwds_i_hi.eq(rwds["IN_LO"]),
            phy_if.rwds_i_lo.eq(rwds["IN_HI"]),
        ]

        # Capture clock dynamic phase shift.
        self.phase_steps = 0
        if shift_pads is not None:
            self.add_phase_shift(shift_pads, shift_sel)

    def add_phase_shift(self, shift_pads, shift_sel, with_csr=True):
        """Capture clock phase control through the PLL dynamic phase shift.

        `phase` is applied on `phase_load` (SHIFT value held, SHIFT_ENA pulsed). Each step delays the
        capture clock by half a period of the PLL post-divider clock.
        """
        self.phase_steps = 2**len(shift_pads["shift"])
        self.phase       = Signal(len(shift_pads["shift"]))
        self.phase_load  = Signal()

        # # #

        phase     = Signal(len(self.phase))
        shift_ena = Signal()
        timer     = Signal(4)
        self.sync += [
            If(self.phase_load,
                phase.eq(self.phase),
                timer.eq(15),
            ).Elif(timer != 0,
                timer.eq(timer - 1),
            ),
            shift_ena.eq((timer > 4) & (timer < 12)),
        ]
        self.comb += [
            shift_pads["shift"].eq(phase),
            shift_pads["shift_sel"].eq(shift_sel),
            shift_pads["shift_ena"].eq(shift_ena),
        ]

# Read Calibration ---------------------------------------------------------------------------------

def hyperram_ddr_best_phase(passes, nphases):
    """Center of the largest (circular) window of passing phases, None if none passes."""
    if passes == 0:
        return None
    if passes == (2**nphases - 1):
        return 0
    best_len, best_start = 0, 0
    for start in range(nphases):
        # Window starts on a passing phase preceded by a failing one.
        if not (passes >> start) & 1 or (passes >> ((start - 1) % nphases)) & 1:
            continue
        length = 0
        while (passes >> ((start + length) % nphases)) & 1:
            length += 1
        if length > best_len:
            best_len, best_start = length, start
    return (best_start + (best_len - 1)//2) % nphases

class HyperRAMDDRCalibration(LiteXModule):
    """Read capture phase calibration (gateware).

    Writes a pattern at `address`, reads it back for each capture phase and selects the center of
    the largest window of passing phases. Memory at `address` is overwritten.
    """
    def __init__(self, bus, phy, address=0x0, words=16, settle_cycles=256):
        nphases     = phy.phase_steps
        dw          = len(bus.w.data)
        self.start  = Signal()
        self.done   = Signal()
        self.error  = Signal()
        self.passes = Signal(nphases)
        self.phase  = Signal(max=nphases)

        # # #

        patterns = [0x00000000, 0xffffffff, 0xaaaa5555, 0x5555aaaa, 0x0001fffe, 0xfffe0001, 0x12345678,
            0x9abcdef0, 0x0f0f0f0f, 0xf0f0f0f0, 0x00ff00ff, 0xff00ff00, 0x13579bdf, 0x2468ace0,
            0x5a5aa5a5, 0xa5a55a5a]
        def pattern(i):
            return Cat(*[Constant(patterns[(i + j) % len(patterns)], 32) for j in range(dw//32)])

        count  = Signal(max=max(words, settle_cycles) + 1)
        index  = Signal(max=nphases + 1)
        ok     = Signal()
        expect = Signal(dw)
        self.comb += Case(count, {i: expect.eq(pattern(i)) for i in range(words)})

        best = Signal(max=nphases)
        self.comb += Case(self.passes, {
            m: best.eq(hyperram_ddr_best_phase(m, nphases) or 0) for m in range(2**nphases)})

        self.comb += [
            bus.aw.addr.eq(address),
            bus.aw.len.eq(words - 1),
            bus.aw.size.eq(log2_int(dw//8)),
            bus.aw.burst.eq(axi.BURST_INCR),
            bus.ar.addr.eq(address),
            bus.ar.len.eq(words - 1),
            bus.ar.size.eq(log2_int(dw//8)),
            bus.ar.burst.eq(axi.BURST_INCR),
            bus.w.data.eq(expect),
            bus.w.strb.eq(2**(dw//8) - 1),
            bus.w.last.eq(count == (words - 1)),
        ]

        self.fsm = fsm = FSM(reset_state="IDLE")
        fsm.act("IDLE",
            If(self.start,
                NextValue(self.done,   0),
                NextValue(self.error,  0),
                NextValue(self.passes, 0),
                NextValue(index, 0),
                NextValue(count, 0),
                NextState("WRITE-AW"),
            )
        )
        fsm.act("WRITE-AW",
            bus.aw.valid.eq(1),
            If(bus.aw.ready, NextState("WRITE-W"))
        )
        fsm.act("WRITE-W",
            bus.w.valid.eq(1),
            If(bus.w.ready,
                NextValue(count, count + 1),
                If(bus.w.last,
                    NextValue(count, 0),
                    NextState("WRITE-B"),
                )
            )
        )
        fsm.act("WRITE-B",
            bus.b.ready.eq(1),
            If(bus.b.valid, NextState("PHASE-LOAD"))
        )
        fsm.act("PHASE-LOAD",
            phy.phase.eq(index),
            phy.phase_load.eq(1),
            NextValue(count, 0),
            NextState("PHASE-SETTLE"),
        )
        fsm.act("PHASE-SETTLE",
            NextValue(count, count + 1),
            If(count == (settle_cycles - 1),
                NextValue(count, 0),
                NextValue(ok, 1),
                NextState("READ-AR"),
            )
        )
        fsm.act("READ-AR",
            bus.ar.valid.eq(1),
            If(bus.ar.ready, NextState("READ-R"))
        )
        fsm.act("READ-R",
            bus.r.ready.eq(1),
            If(bus.r.valid,
                NextValue(count, count + 1),
                If((bus.r.data != expect) | (bus.r.resp != axi.RESP_OKAY),
                    NextValue(ok, 0),
                ),
                If(bus.r.last,
                    NextState("READ-DONE"),
                )
            )
        )
        fsm.act("READ-DONE",
            NextValue(self.passes, self.passes | (ok << index)),
            NextValue(index, index + 1),
            If(index == (nphases - 1),
                NextState("SELECT"),
            ).Else(
                NextState("PHASE-LOAD"),
            )
        )
        fsm.act("SELECT",
            NextValue(self.phase, best),
            NextValue(self.error, self.passes == 0),
            phy.phase.eq(best),
            phy.phase_load.eq(1),
            NextValue(count, 0),
            NextState("SELECT-SETTLE"),
        )
        fsm.act("SELECT-SETTLE",
            NextValue(count, count + 1),
            If(count == (settle_cycles - 1),
                NextValue(self.done, 1),
                NextState("IDLE"),
            )
        )

# HyperRAM DDR -------------------------------------------------------------------------------------

class HyperRAMDDR(LiteXModule):
    """HyperRAM DDR controller (PHY + core, optional read capture calibration).

    With a PHY providing a capture phase control, the read capture phase is calibrated at startup
    (and on `calibrate` CSR writes); the AXI bus is held off until calibration is done.
    """
    def __init__(self, phy, clk_freq, with_calibration=True, with_csr=True, **kwargs):
        self.phy  = phy
        self.core = core = HyperRAMDDRCore(phy.phy_if, clk_freq=clk_freq, with_csr=with_csr, **kwargs)
        self.bus  = bus  = axi.AXIInterface(
            data_width    = core.bus.data_width,
            address_width = 32,
            id_width      = len(core.bus.aw.id),
        )

        # # #

        if not (with_calibration and getattr(phy, "phase_steps", 0)):
            self.comb += bus.connect(core.bus)
            return

        cal_bus  = axi.AXIInterface(data_width=core.bus.data_width, address_width=32, id_width=len(core.bus.aw.id))
        self.cal = cal = HyperRAMDDRCalibration(cal_bus, phy)

        # Start calibration once the core is ready (and on CSR request).
        cal_req   = Signal(reset=1)
        cal_start = Signal()
        self.comb += cal_start.eq(cal_req & core.ready & cal.fsm.ongoing("IDLE"))
        self.sync += If(cal_start, cal_req.eq(0))
        self.comb += cal.start.eq(cal_start)
        busy = Signal(reset=1)
        self.sync += [
            If(cal_start,
                busy.eq(1),
            ).Elif(cal.done,
                busy.eq(0),
            )
        ]
        self.comb += If(busy,
            cal_bus.connect(core.bus),
        ).Else(
            bus.connect(core.bus),
        )

        if with_csr:
            self._calibration = CSRStatus(fields=[
                CSRField("done",   size=1, offset=0,  description="Calibration done."),
                CSRField("error",  size=1, offset=1,  description="Calibration failed (no passing phase)."),
                CSRField("phase",  size=8, offset=8,  description="Selected capture phase."),
                CSRField("passes", size=8, offset=16, description="Passing capture phases (one bit per phase)."),
            ])
            self._calibrate = CSR()
            self.comb += [
                self._calibration.fields.done.eq(~busy & cal.done),
                self._calibration.fields.error.eq(cal.error),
                self._calibration.fields.phase.eq(cal.phase),
                self._calibration.fields.passes.eq(cal.passes),
            ]
            self.sync += If(self._calibrate.re, cal_req.eq(1))
