import itertools
import pathlib
import sys
import dataclasses
from typing import Literal

src_path = pathlib.Path(__file__).parent.parent / 'magic_state_cultivation' / 'upstream' / 'src'
assert src_path.exists()
sys.path.append(str(src_path))

import cultiv
import gen
import stim
from cultiv._construction._cultivation_stage import (
    make_inject_and_cultivate_chunks_d3,
    make_inject_and_cultivate_chunks_d5,
)
from cultiv._construction._escape_stage import (
    make_color_code_to_big_matchable_code_escape_chunks,
)


# Supported circuit types:
supported_circuit_types = {
    'escape-to-big-matchable-code',
    'idle-matchable-code',
    'surface-code-memory',
    'inject+cultivate',
    'end2end-inplace-distillation',
    'escape-to-big-color-code',
    'surface-code-cnot'
}

# Supported injection protocols:
supported_injection_protocols = {
    'degenerate',
    'bell',
    'unitary'
}

# Supported noise models:
supported_noise_models = {
    'circuit-level-SI1000',
    'uniform-depolarizing'
}

# The data class for caontaining the parameters for generating a circuit.
@dataclasses.dataclass
class CircuitGenParams:
    circuit_type: str
    noise_model: str
    noise_strength: float
    gateset: str
    basis: str
    num_layers: int
    injection_protocol: str | None = None
    r_in_escape: int | None = None
    d1: int | None = None
    r_post_escape: int | None = None
    d2: int | None = None
    v: int | None = None
    StageTimelineMap: dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)



# The class for generating circuits based on the parameters.
class CircuitGenerator:
    def __init__(self, params: CircuitGenParams, HasNoise: bool = True):
        self.params = params
        self.ideal_circuit = None
        self.noisy_circuit = None
        self.HasNoise = HasNoise

    @staticmethod
    def _count_ticks_in_stim_circuit(circuit: stim.Circuit) -> int:
        return sum(1 for inst in circuit if inst.name == 'TICK')

    def _count_ticks_in_chunk_like(self, chunk_like: object) -> int:
        if hasattr(chunk_like, 'repetitions') and hasattr(chunk_like, 'chunks'):
            repetitions = int(getattr(chunk_like, 'repetitions'))
            chunks = list(getattr(chunk_like, 'chunks'))
            return repetitions * sum(self._count_ticks_in_chunk_like(c) for c in chunks)
        circuit = gen.compile_chunks_into_circuit([chunk_like], add_mpp_boundaries=True)
        return self._count_ticks_in_stim_circuit(circuit)

    @staticmethod
    def _append_stage(
        timeline: dict[str, tuple[int, int]],
        stage_name: str,
        start_t: int,
        length: int,
    ) -> int:
        if length <= 0:
            return start_t
        end_t = start_t + length - 1
        timeline[stage_name] = (start_t, end_t)
        return end_t + 1

    @staticmethod
    def _chunk_stage_tag(chunk_like: object) -> str | None:
        if not hasattr(chunk_like, 'flows'):
            return None
        flows = getattr(chunk_like, 'flows')
        for flow in flows:
            flags = getattr(flow, 'flags', ())
            for f in flags:
                if f.startswith('stage='):
                    return f[len('stage='):]
        return None

    def _build_stage_timeline_map(self) -> dict[str, tuple[int, int]]:
        timeline: dict[str, tuple[int, int]] = {}
        t = 0

        def scaled_lengths(total: int, weighted_parts: list[tuple[str, int]]) -> list[tuple[str, int]]:
            parts = [(name, max(0, w)) for name, w in weighted_parts]
            s = sum(w for _, w in parts)
            if total <= 0 or s <= 0:
                return [(name, 0) for name, _ in parts]
            raw = [(name, total * w / s) for name, w in parts]
            ints = [(name, int(v)) for name, v in raw]
            used = sum(v for _, v in ints)
            rem = total - used
            frac_order = sorted(
                range(len(raw)),
                key=lambda i: raw[i][1] - int(raw[i][1]),
                reverse=True,
            )
            ints = ints[:]
            for i in frac_order[:rem]:
                name, v = ints[i]
                ints[i] = (name, v + 1)
            return ints

        def inject_cultivate_semantic_parts() -> list[tuple[str, int]]:
            if self.params.d1 == 3:
                chunks = make_inject_and_cultivate_chunks_d3(style=self.params.injection_protocol or 'unitary')
            elif self.params.d1 == 5:
                chunks = make_inject_and_cultivate_chunks_d5(style=self.params.injection_protocol or 'unitary')
            else:
                chunks = []
            inject_ticks = 0
            cultivate_ticks = 0
            for i, chunk in enumerate(chunks):
                ticks = self._count_ticks_in_chunk_like(chunk)
                tag = self._chunk_stage_tag(chunk)
                if tag in ('inject', 'injection'):
                    inject_ticks += ticks
                elif tag == 'cultivation':
                    cultivate_ticks += ticks
                else:
                    if i == 0:
                        inject_ticks += ticks
                    else:
                        cultivate_ticks += ticks
            return [('injection', inject_ticks), ('cultivation', cultivate_ticks)]

        def escape_semantic_parts() -> list[tuple[str, int]]:
            chunks = make_color_code_to_big_matchable_code_escape_chunks(
                dcolor=self.params.d1,
                dsurface=self.params.d2,
                basis=self.params.basis,
                r_growing=self.params.r_in_escape,
                r_end=self.params.r_post_escape,
            )
            return [
                ('code-grow', sum(self._count_ticks_in_chunk_like(c) for c in chunks[:2])),
                ('escape', sum(self._count_ticks_in_chunk_like(c) for c in chunks[2:5])),
                ('post-escape', self._count_ticks_in_chunk_like(chunks[5])),
            ]

        if self.params.circuit_type == 'inject+cultivate':
            total_ticks = self._count_ticks_in_stim_circuit(self.ideal_circuit)
            for stage_name, length in scaled_lengths(total_ticks, inject_cultivate_semantic_parts()):
                t = self._append_stage(timeline, stage_name, t, length)
            return timeline

        if self.params.circuit_type == 'escape-to-big-matchable-code':
            total_ticks = self._count_ticks_in_stim_circuit(self.ideal_circuit)
            for stage_name, length in scaled_lengths(total_ticks, escape_semantic_parts()):
                t = self._append_stage(timeline, stage_name, t, length)
            return timeline

        if self.params.circuit_type == 'end2end-inplace-distillation':
            total_ticks = self._count_ticks_in_stim_circuit(self.ideal_circuit)
            inject_cult_ticks = self._count_ticks_in_stim_circuit(cultiv.make_inject_and_cultivate_circuit(
                dcolor=self.params.d1,
                inject_style=self.params.injection_protocol if self.params.injection_protocol is not None else 'unitary',
                basis=self.params.basis,
            ))
            escape_only_ticks = self._count_ticks_in_stim_circuit(cultiv.make_escape_to_big_matchable_code_circuit(
                dcolor=self.params.d1,
                dsurface=self.params.d2,
                basis=self.params.basis,
                r_growing=self.params.r_in_escape,
                r_end=self.params.r_post_escape,
            ))
            overlap = max(0, inject_cult_ticks + escape_only_ticks - total_ticks)
            inject_cult_ticks_in_end2end = max(0, inject_cult_ticks - overlap)
            escape_ticks_in_end2end = max(0, total_ticks - inject_cult_ticks_in_end2end)

            for stage_name, length in scaled_lengths(inject_cult_ticks_in_end2end, inject_cultivate_semantic_parts()):
                t = self._append_stage(timeline, stage_name, t, length)
            for stage_name, length in scaled_lengths(escape_ticks_in_end2end, escape_semantic_parts()):
                t = self._append_stage(timeline, stage_name, t, length)
            return timeline

        if self.params.circuit_type in {'idle-matchable-code', 'surface-code-memory'}:
            total_ticks = self._count_ticks_in_stim_circuit(self.ideal_circuit)
            t = self._append_stage(timeline, 'memory-idle', t, total_ticks)
            return timeline

        if self.params.circuit_type == 'escape-to-big-color-code':
            total_ticks = self._count_ticks_in_stim_circuit(self.ideal_circuit)
            for stage_name, length in scaled_lengths(total_ticks, [
                ('code-grow', 1),
                ('post-grow-idle', self.params.r_post_escape or 0),
            ]):
                t = self._append_stage(timeline, stage_name, t, length)
            return timeline

        if self.params.circuit_type == 'surface-code-cnot':
            total_ticks = self._count_ticks_in_stim_circuit(self.ideal_circuit)
            t = self._append_stage(timeline, 'lattice-surgery-cnot', t, total_ticks)
            return timeline

        return timeline
    
    @staticmethod
    def from_input_params(basis: Literal['X', 'Y', 'Z', 'EPR'], gateset: str, \
        circuit_type: str, noise_model: str, noise_strength: float, injection_protocol: str | None = None, r_in_escape: int | None = None, d1: int | None = None, \
            r_post_escape: int | None = None, d2: int | None = None, v: int | None = None, HasNoise: bool = True) -> 'CircuitGenerator':
        if circuit_type not in supported_circuit_types:
            raise ValueError(f'Unsupported circuit type: {circuit_type}')
        if injection_protocol is not None and injection_protocol not in supported_injection_protocols:
            raise ValueError(f'Unsupported injection protocol: {injection_protocol}')
        if noise_model not in supported_noise_models:
            raise ValueError(f'Unsupported noise model: {noise_model}')
        params = CircuitGenParams(
            circuit_type=circuit_type,
            injection_protocol=injection_protocol,
            noise_model=noise_model,
            noise_strength=noise_strength,
            gateset=gateset,
            basis=basis,
            num_layers=0, # This will be set later based on the circuit type and other parameters.
            r_in_escape=r_in_escape,
            d1=d1,
            r_post_escape=r_post_escape,
            d2=d2,
            v=v,
        )
        return CircuitGenerator(params, HasNoise=HasNoise)
    

    def generate(self) -> gen.LayerCircuit:
        circuit = None
        if self.params.circuit_type == 'inject+cultivate':
            if self.params.injection_protocol is None:
                raise ValueError('Injection protocol must be specified for inject+cultivate circuit type.')
            elif self.params.injection_protocol == 'degenerate':
                circuit = cultiv.make_inject_and_cultivate_circuit(inject_style='degenerate', dcolor=self.params.d1, basis=self.params.basis)
            elif self.params.injection_protocol == 'bell':
                circuit = cultiv.make_inject_and_cultivate_circuit(inject_style='bell', dcolor=self.params.d1, basis=self.params.basis)
            elif self.params.injection_protocol == 'unitary':
                circuit = cultiv.make_inject_and_cultivate_circuit(inject_style='unitary', dcolor=self.params.d1, basis=self.params.basis)
            else:
                raise NotImplementedError(f'{self.params.injection_protocol=}')
            self.params.r_in_escape = None
            self.params.r_post_escape = None
            self.params.d2 = None
            self.params.v = None
        elif self.params.circuit_type == 'escape-to-big-matchable-code':
            circuit = cultiv.make_escape_to_big_matchable_code_circuit(
                dcolor=self.params.d1,
                dsurface=self.params.d2,
                basis=self.params.basis,
                r_growing=self.params.r_in_escape,
                r_end=self.params.r_post_escape,
            )
            self.params.v = None
        elif self.params.circuit_type == 'idle-matchable-code':
            circuit = cultiv.make_idle_matchable_code_circuit(dcolor=self.params.d1, dsurface=self.params.d2, basis=self.params.basis, rounds=self.params.r_post_escape)
            self.params.r_in_escape = None
            self.params.v = None
        elif self.params.circuit_type == 'surface-code-memory':
            circuit = cultiv.make_surface_code_memory_circuit(dsurface=self.params.d2, basis=self.params.basis, rounds=self.params.r_post_escape)
            self.params.r_in_escape = None
            self.params.d1 = None
            self.params.v = None
        elif self.params.circuit_type == 'end2end-inplace-distillation':
            circuit = cultiv.make_end2end_cultivation_circuit(
                dcolor=self.params.d1,
                dsurface=self.params.d2,
                basis=self.params.basis,
                r_growing=self.params.r_in_escape,
                r_end=self.params.r_post_escape,
                inject_style=self.params.injection_protocol if self.params.injection_protocol is not None else 'unitary',
            )
            self.params.v = None
        elif self.params.circuit_type == 'escape-to-big-color-code':
            circuit = cultiv.make_escape_to_big_color_code_circuit(
                start_width=self.params.d1,
                end_width=self.params.d2,
                rounds=self.params.r_post_escape,
                basis=self.params.basis,
            )
            self.params.r_in_escape = None
            self.params.v = None
        elif self.params.circuit_type == 'surface-code-cnot':
            circuit = cultiv.make_surface_code_cnot(
                distance=self.params.d2,
                basis=self.params.basis,
            )
            self.params.r_in_escape = None
            self.params.v = None
            self.params.r_post_escape = None
            self.params.d1 = None
        else:
            raise NotImplementedError(f'{self.params.circuit_type=}')
        
        self.ideal_circuit = circuit
        self.params.num_layers = self._count_ticks_in_stim_circuit(self.ideal_circuit)
        self.params.StageTimelineMap = self._build_stage_timeline_map()
        if self.HasNoise:
            if self.params.gateset == 'cz':
                if self.params.noise_model != 'circuit-level-SI1000':
                    raise ValueError(f'Noise model must be circuit-level-SI1000 for cz gateset.')
                noise = 'si1000'
                # Match tools/make_circuits: transpile before applying SI1000 for CZ gateset.
                self.ideal_circuit = gen.transpile_to_z_basis_interaction_circuit(self.ideal_circuit)
                noise_model = gen.NoiseModel.si1000(self.params.noise_strength)
            else:
                if self.params.noise_model == 'circuit-level-SI1000':
                    noise = 'si1000'
                    noise_model = gen.NoiseModel.si1000(self.params.noise_strength)
                elif self.params.noise_model == 'uniform-depolarizing':
                    noise = 'uniform'
                    noise_model = gen.NoiseModel.uniform_depolarizing(self.params.noise_strength)
                else:
                    raise ValueError(f'Unsupported noise model: {self.params.noise_model}')
            self.noisy_circuit = noise_model.noisy_circuit_skipping_mpp_boundaries(self.ideal_circuit)
        else:
            self.noisy_circuit = None
        return
