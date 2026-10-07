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
    injection_protocol: str | None = None
    noise_model: str
    noise_strength: float
    gateset: str
    basis: str
    num_layers: int
    r_in_escape: int | None = None
    d1: int | None = None
    r_post_escape: int | None = None
    d2: int | None = None
    v: int | None = None



# The class for generating circuits based on the parameters.
class CircuitGenerator:
    def __init__(self, params: CircuitGenParams, HasNoise: bool = True):
        self.params = params
        self.ideal_circuit = None
        self.noisy_circuit = None
        self.HasNoise = HasNoise
    
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
        if self.HasNoise:
            if self.params.gateset == 'cz':
                if self.params.noise_model != 'circuit-level-SI1000':
                    raise ValueError(f'Noise model must be circuit-level-SI1000 for cz gateset.')
                noise = 'si1000'
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