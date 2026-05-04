from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import statistics
import matplotlib.pyplot as plt

from .homogenization import LinearHomogenizationResult

from homicsx.utils import (
    extract_effective_moduli_3d,
    extract_effective_moduli_2d_plane_strain,
)


@dataclass
class EnsembleStatSummary:
    """
    Container for summary of results of an ensemble study.

    Attributes
    ----------
    engineering_moduli_mean: dict[str, float]
        keys: "E" for Yong modulus, "nu" for Poison ratio, "kappa" for bulk modulus, and "mu" for shear modulus.
    engineering_moduli_variance: dict[str, float]
        keys: "E" for Yong modulus, "nu" for Poison ratio, "kappa" for bulk modulus, and "mu" for shear modulus.
    engineering_moduli_stdev: dict[str, float]
        keys: "E" for Yong modulus, "nu" for Poison ratio, "kappa" for bulk modulus, and "mu" for shear modulus.
    engineering_moduli_max: dict[str, float]
        keys: "E" for Yong modulus, "nu" for Poison ratio, "kappa" for bulk modulus, and "mu" for shear modulus.
    engineering_moduli_min: dict[str, float]
        keys: "E" for Yong modulus, "nu" for Poison ratio, "kappa" for bulk modulus, and "mu" for shear modulus.
    C_hom_component_avg: np.ndarray
        Component-wise mean of homogenized stiffness tensor.
    C_hom_component_variance: np.ndarray
        Component-wise variance of homogenized stiffness tensor. Defined if size of ensemble study is larger than 2.
    C_hom_component_stdev: np.ndarray
        Component-wise standard-deviation of homogenized stiffness tensor. Defined if size of ensemble study is larger than 2.
    C_hom_component_max: np.ndarray
        Component-wise max of homogenized stiffness tensor.
    C_hom_component_min: np.ndarray
        Component-wise min of homogenized stiffness tensor.
    """
    engineering_moduli_mean: dict[str, float]
    engineering_moduli_variance: dict[str, float]
    engineering_moduli_stdev: dict[str, float]
    engineering_moduli_max: dict[str, float]
    engineering_moduli_min: dict[str, float]
    C_hom_component_avg: np.ndarray
    C_hom_component_variance: np.ndarray
    C_hom_component_stdev: np.ndarray
    C_hom_component_max: np.ndarray
    C_hom_component_min: np.ndarray


@dataclass
class EnsembleStudyResult:
    """
    Container for ensemble of linear study results, with built in summarizing and visualization utilities.

    Attributes
    ----------
    result_list: list[`HomogenizationResult`]
        Contains the results of the ensemble study in its entirety.
    result_summary: `EnsembleStatSummary`
        Contains the summary of the ensemble study.
        
    See also
    --------
    EnsembleStatSummary
    """
    result_list: list[LinearHomogenizationResult]
    metadata: dict[str, any] = field(default_factory=dict)


    def _calculate_ensemble_statistics(
        self,
        ensumble_result: list[LinearHomogenizationResult],
    ) -> EnsembleStatSummary:
        homogenization_dim = ensumble_result[0].metadata["dim"]
        two_dimensional_formulation = ensumble_result[0].metadata["two_dimensional_formulation"]

        if homogenization_dim==3:
            C_hom_list = np.ndarray((6, 6, len(ensumble_result)))
            kappa_list = []
            mu_list = []
            E_list = []
            nu_list = []

            for i, hom_result in enumerate(ensumble_result):
                kappa, mu, E, nu = extract_effective_moduli_3d(hom_result.C_hom)

                kappa_list.append(kappa)
                mu_list.append(mu)
                E_list.append(E)
                nu_list.append(nu)

                C_hom_list[:, :, i] = hom_result.C_hom

            engineering_moduli_mean = {
                "kappa": [],
                "mu": [],
                "E": [],
                "nu": [],
            }
            engineering_moduli_mean["kappa"] = statistics.mean(kappa_list)
            engineering_moduli_mean["mu"] = statistics.mean(mu_list)
            engineering_moduli_mean["E"] = statistics.mean(E_list)
            engineering_moduli_mean["nu"] = statistics.mean(nu_list)

            engineering_moduli_variance = {
                "kappa": [],
                "mu": [],
                "E": [],
                "nu": [],
            }
            if len(ensumble_result)>=2:
                engineering_moduli_variance["kappa"] = statistics.variance(kappa_list)
                engineering_moduli_variance["mu"] = statistics.variance(mu_list)
                engineering_moduli_variance["E"] = statistics.variance(E_list)
                engineering_moduli_variance["nu"] = statistics.variance(nu_list)

            engineering_moduli_stdev = {
                "kappa": [],
                "mu": [],
                "E": [],
                "nu": [],
            }
            if len(ensumble_result)>=2:
                engineering_moduli_stdev["kappa"] = statistics.stdev(kappa_list)
                engineering_moduli_stdev["mu"] = statistics.stdev(mu_list)
                engineering_moduli_stdev["E"] = statistics.stdev(E_list)
                engineering_moduli_stdev["nu"] = statistics.stdev(nu_list)

            engineering_moduli_max = {
                "kappa": [],
                "mu": [],
                "E": [],
                "nu": [],
            }
            engineering_moduli_max["kappa"] = max(kappa_list)
            engineering_moduli_max["mu"] = max(mu_list)
            engineering_moduli_max["E"] = max(E_list)
            engineering_moduli_max["nu"] = max(nu_list)

            engineering_moduli_min = {
                "kappa": [],
                "mu": [],
                "E": [],
                "nu": [],
            }
            engineering_moduli_min["kappa"] = min(kappa_list)
            engineering_moduli_min["mu"] = min(mu_list)
            engineering_moduli_min["E"] = min(E_list)
            engineering_moduli_min["nu"] = min(nu_list)

            C_hom_component_avg = np.zeros((6, 6))
            C_hom_component_variance = np.zeros((6, 6))
            C_hom_component_stdev = np.zeros((6, 6))
            C_hom_component_max = np.zeros((6, 6))
            C_hom_component_min = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    C_hom_component_avg[i, j] = statistics.mean(C_hom_list[i, j, :])
                    if len(ensumble_result)>=2:
                        C_hom_component_variance[i, j] = statistics.variance(C_hom_list[i, j, :])
                        C_hom_component_stdev[i, j] = statistics.stdev(C_hom_list[i, j, :])
                    C_hom_component_max[i, j] = max(C_hom_list[i, j, :])
                    C_hom_component_min[i, j] = min(C_hom_list[i, j, :])

        
        elif homogenization_dim==2:
            if two_dimensional_formulation==None:
                raise ValueError('Two dimensional formulation of the fem problem must be either "plane_strain" or "plane_stress". Recieverd "None".')
            
            elif two_dimensional_formulation=="plane_strain":
                C_hom_list = np.ndarray((3, 3, len(ensumble_result)))
                kappa_list = []
                mu_list = []
                E_list = []
                nu_list = []

                for i, hom_result in enumerate(ensumble_result):
                    kappa, mu, E, nu = extract_effective_moduli_2d_plane_strain(hom_result.C_hom)

                    kappa_list.append(kappa)
                    mu_list.append(mu)
                    E_list.append(E)
                    nu_list.append(nu)

                    C_hom_list[:, :, i] = hom_result.C_hom

                engineering_moduli_mean = {
                    "kappa": [],
                    "mu": [],
                    "E": [],
                    "nu": [],
                }
                engineering_moduli_mean["kappa"] = statistics.mean(kappa_list)
                engineering_moduli_mean["mu"] = statistics.mean(mu_list)
                engineering_moduli_mean["E"] = statistics.mean(E_list)
                engineering_moduli_mean["nu"] = statistics.mean(nu_list)

                engineering_moduli_variance = {
                    "kappa": [],
                    "mu": [],
                    "E": [],
                    "nu": [],
                }
                if len(ensumble_result)>=2:
                    engineering_moduli_variance["kappa"] = statistics.variance(kappa_list)
                    engineering_moduli_variance["mu"] = statistics.variance(mu_list)
                    engineering_moduli_variance["E"] = statistics.variance(E_list)
                    engineering_moduli_variance["nu"] = statistics.variance(nu_list)

                engineering_moduli_stdev = {
                    "kappa": [],
                    "mu": [],
                    "E": [],
                    "nu": [],
                }
                if len(ensumble_result)>=2:
                    engineering_moduli_stdev["kappa"] = statistics.stdev(kappa_list)
                    engineering_moduli_stdev["mu"] = statistics.stdev(mu_list)
                    engineering_moduli_stdev["E"] = statistics.stdev(E_list)
                    engineering_moduli_stdev["nu"] = statistics.stdev(nu_list)

                engineering_moduli_max = {
                    "kappa": [],
                    "mu": [],
                    "E": [],
                    "nu": [],
                }
                engineering_moduli_max["kappa"] = max(kappa_list)
                engineering_moduli_max["mu"] = max(mu_list)
                engineering_moduli_max["E"] = max(E_list)
                engineering_moduli_max["nu"] = max(nu_list)

                engineering_moduli_min = {
                    "kappa": [],
                    "mu": [],
                    "E": [],
                    "nu": [],
                }
                engineering_moduli_min["kappa"] = min(kappa_list)
                engineering_moduli_min["mu"] = min(mu_list)
                engineering_moduli_min["E"] = min(E_list)
                engineering_moduli_min["nu"] = min(nu_list)

                C_hom_component_avg = np.zeros((3, 3))
                C_hom_component_variance = np.zeros((3, 3))
                C_hom_component_stdev = np.zeros((3, 3))
                C_hom_component_max = np.zeros((3, 3))
                C_hom_component_min = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        C_hom_component_avg[i, j] = statistics.mean(C_hom_list[i, j, :])
                        if len(ensumble_result)>=2:
                            C_hom_component_variance[i, j] = statistics.variance(C_hom_list[i, j, :])
                            C_hom_component_stdev[i, j] = statistics.stdev(C_hom_list[i, j, :])
                        C_hom_component_max[i, j] = max(C_hom_list[i, j, :])
                        C_hom_component_min[i, j] = min(C_hom_list[i, j, :])

            elif two_dimensional_formulation=="plane_stress":
                raise NotImplementedError("In the current development phase, plane stress homogenization is not inplemented yet.")
            
    
        output = EnsembleStatSummary(
            C_hom_component_avg=C_hom_component_avg,
            C_hom_component_variance=C_hom_component_variance,
            C_hom_component_stdev=C_hom_component_stdev,
            C_hom_component_max=C_hom_component_max,
            C_hom_component_min=C_hom_component_min,
            engineering_moduli_mean=engineering_moduli_mean,
            engineering_moduli_variance=engineering_moduli_variance,
            engineering_moduli_stdev=engineering_moduli_stdev,
            engineering_moduli_max=engineering_moduli_max,
            engineering_moduli_min=engineering_moduli_min,
            )
        
        return output
    

    def __post_init__(self) -> None:
        self.result_summary: EnsembleStatSummary = self._calculate_ensemble_statistics(
            self.result_list
        )


    def result_moduli(self) -> dict[str, list[float]]:
        """Returns result moduli as a dict with keys "E", "nu", "kappa", "mu", and list of values as values."""
        homogenization_dim = self.result_list[0].metadata["dim"]
        two_dimensional_formulation = self.result_list[0].metadata["two_dimensional_formulation"]

        if homogenization_dim==3:
            kappa_list = []
            mu_list = []
            E_list = []
            nu_list = []

            for hom_result in self.result_list:
                kappa, mu, E, nu = extract_effective_moduli_3d(hom_result.C_hom)

                kappa_list.append(kappa)
                mu_list.append(mu)
                E_list.append(E)
                nu_list.append(nu)
        
        elif homogenization_dim==2:
            if two_dimensional_formulation==None:
                raise ValueError('Two dimensional formulation of the fem problem must be either "plane_strain" or "plane_stress". Recieverd "None".')
            
            elif two_dimensional_formulation=="plane_strain":
                kappa_list = []
                mu_list = []
                E_list = []
                nu_list = []

                for hom_result in self.result_list:
                    kappa, mu, E, nu = extract_effective_moduli_2d_plane_strain(hom_result.C_hom)

                    kappa_list.append(kappa)
                    mu_list.append(mu)
                    E_list.append(E)
                    nu_list.append(nu)
            
            elif two_dimensional_formulation=="plane_stress":
                raise NotImplementedError("In the current development phase, plane stress homogenization is not inplemented yet.")
        
        return {
            'kappa': kappa_list,
            'mu': mu_list,
            'E': E_list,
            'nu': nu_list,
        }


    def print_summary(self) -> None:
        """
        Prints the auto-generated summary of the ensemble study results.
        """
        print('=========================================================')
        print('---------------- Ensemble Study Summary -----------------')
        print('C_hom component statistics:')
        print('Mean:')
        with np.printoptions(suppress=True, precision=3):
            print(self.result_summary.C_hom_component_avg)
        print('\n')
        print('Variance:')
        with np.printoptions(suppress=True, precision=3):
            print(self.result_summary.C_hom_component_variance)
        print('\n')
        print('Standard deviation:')
        with np.printoptions(suppress=True, precision=3):
            print(self.result_summary.C_hom_component_stdev)
        print('\n')
        print('Max:')
        with np.printoptions(suppress=True, precision=3):
            print(self.result_summary.C_hom_component_max)
        print('\n')
        print('Min:')
        with np.printoptions(suppress=True, precision=3):
            print(self.result_summary.C_hom_component_min)
        
        print('\n')
        print('Engineering moduli statistics: ')
        print('Mean: ')
        print(f'    kappa:  {self.result_summary.engineering_moduli_mean["kappa"]}')
        print(f'    mu:     {self.result_summary.engineering_moduli_mean["mu"]}')
        print(f'    E:      {self.result_summary.engineering_moduli_mean["E"]}')
        print(f'    nu:     {self.result_summary.engineering_moduli_mean["nu"]}')
        print('\n')
        print('Variance: ')
        print(f'    kappa:  {self.result_summary.engineering_moduli_variance["kappa"]}')
        print(f'    mu:     {self.result_summary.engineering_moduli_variance["mu"]}')
        print(f'    E:      {self.result_summary.engineering_moduli_variance["E"]}')
        print(f'    nu:     {self.result_summary.engineering_moduli_variance["nu"]}')
        print('\n')
        print('Standard deviation: ')
        print(f'    kappa:  {self.result_summary.engineering_moduli_stdev["kappa"]}')
        print(f'    mu:     {self.result_summary.engineering_moduli_stdev["mu"]}')
        print(f'    E:      {self.result_summary.engineering_moduli_stdev["E"]}')
        print(f'    nu:     {self.result_summary.engineering_moduli_stdev["nu"]}')
        print('\n')
        print('Max: ')
        print(f'    kappa:  {self.result_summary.engineering_moduli_max["kappa"]}')
        print(f'    mu:     {self.result_summary.engineering_moduli_max["mu"]}')
        print(f'    E:      {self.result_summary.engineering_moduli_max["E"]}')
        print(f'    nu:     {self.result_summary.engineering_moduli_max["nu"]}')
        print('\n')
        print('Min: ')
        print(f'    kappa:  {self.result_summary.engineering_moduli_min["kappa"]}')
        print(f'    mu:     {self.result_summary.engineering_moduli_min["mu"]}')
        print(f'    E:      {self.result_summary.engineering_moduli_min["E"]}')
        print(f'    nu:     {self.result_summary.engineering_moduli_min["nu"]}')
        print('=========================================================')
 

    def visualize_moduli_histogram(
        self, 
        num_bins: int,
    ) -> None:
        """
         Visualizes moduli distribution using histogram charts.
        """
        moduli_list_dict = self.result_moduli()
        E_list = moduli_list_dict['E']
        nu_list = moduli_list_dict['nu']
        kappa_list = moduli_list_dict['kappa']
        mu_list = moduli_list_dict['mu']
        
        fig, axes = plt.subplots(2, 2)

        axes[0][0].hist(E_list, bins=num_bins, edgecolor='black')
        axes[0][0].set_title("Young moduli distribution")
        axes[0][0].set_xlabel("Value")
        axes[0][0].set_ylabel("Frequency")

        axes[0][1].hist(nu_list, bins=num_bins, edgecolor='black')
        axes[0][1].set_title("Poison ration distribution")
        axes[0][1].set_xlabel("Value")
        axes[0][1].set_ylabel("Frequency")

        axes[1][0].hist(kappa_list, bins=num_bins, edgecolor='black')
        axes[1][0].set_title("Bulk moduli distribution")
        axes[1][0].set_xlabel("Value")
        axes[1][0].set_ylabel("Frequency")

        axes[1][1].hist(mu_list, bins=num_bins, edgecolor='black')
        axes[1][1].set_title("Shear moduli distribution")
        axes[1][1].set_xlabel("Value")
        axes[1][1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()


__all__ = [
    # stochastic
    "EnsembleStatSummary",
    "EnsembleStudyResult",
]


