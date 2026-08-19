"""
Reproducible implementation of the maintenance scheduling model.

Dependencies:
    numpy
    pandas
    scipy
    tqdm
    openpyxl
    xlsxwriter

The script:
    1. solves the binary linear maintenance scheduling model;
    2. generates synthetic instances;
    3. performs Monte Carlo simulation;
    4. calculates VaR and CVaR;
    5. exports the computational results to Excel.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from tqdm.auto import tqdm


def fit_exponential_curve(lambda_val, horizon):
    periods = np.arange(1, horizon + 1)

    probability = 1 - np.exp(-lambda_val * periods)

    alpha_val, beta_val = np.polyfit(
        periods,
        probability,
        1
    )

    fitted = alpha_val * periods + beta_val

    error = np.mean(
        np.abs(
            (probability - fitted)
            / np.maximum(probability, 1e-8)
        )
    ) * 100

    return alpha_val, max(0.0, beta_val), error


def solve_model(
    equipment,
    horizon,
    budget,
    labor_capacity,
    minimum_interval=2,
    maximum_interval=5,
    recovery=4.0,
    effectiveness=0.85,
    initial_ages=3.0,
    limit_probability=True,
):

    number_items = len(equipment)
    number_variables = number_items * horizon

    effectiveness_arr = (
        np.full(
            (number_items, horizon),
            float(effectiveness)
        )
        if np.isscalar(effectiveness)
        else np.asarray(effectiveness, dtype=float)
    )

    if effectiveness_arr.ndim == 1:
        effectiveness_arr = np.repeat(
            effectiveness_arr[:, None],
            horizon,
            axis=1
        )

    initial_ages_arr = (
        np.full(number_items, float(initial_ages))
        if np.isscalar(initial_ages)
        else np.asarray(initial_ages, dtype=float)
    )

    def index(item_pos, period_pos):
        return item_pos * horizon + period_pos

    objective_coefficients = np.zeros(number_variables)

    harmonic_tail = np.array([
        np.sum(
            1 / np.arange(period_pos + 1, horizon + 1)
        )
        for period_pos in range(horizon)
    ])

    for item_pos, (_, item) in enumerate(equipment.iterrows()):

        for period_pos in range(horizon):

            objective_coefficients[
                index(item_pos, period_pos)
            ] = (
                -recovery
                * effectiveness_arr[item_pos, period_pos]
                * item["alpha"]
                * item["failure_cost"]
                * harmonic_tail[period_pos]
            )

    rows = []
    lower_bounds = []
    upper_bounds = []

    # ----------------------------------------------------------
    # Resource constraints
    # ----------------------------------------------------------

    for period_pos in range(horizon):

        cost_row = np.zeros(number_variables)
        labor_row = np.zeros(number_variables)

        for item_pos, (_, item) in enumerate(
            equipment.iterrows()
        ):

            cost_row[
                index(item_pos, period_pos)
            ] = item["pm_cost"]

            labor_row[
                index(item_pos, period_pos)
            ] = item["labor"]

        rows.extend([
            cost_row,
            labor_row
        ])

        lower_bounds.extend([
            -np.inf,
            -np.inf
        ])

        upper_bounds.extend([
            budget,
            labor_capacity
        ])

    # ----------------------------------------------------------
    # Minimum maintenance interval
    # ----------------------------------------------------------

    for item_pos in range(number_items):

        for start in range(
            horizon - minimum_interval + 1
        ):

            row = np.zeros(number_variables)

            for period_pos in range(
                start,
                start + minimum_interval
            ):

                row[
                    index(item_pos, period_pos)
                ] = 1

            rows.append(row)

            lower_bounds.append(-np.inf)
            upper_bounds.append(1)

    # ----------------------------------------------------------
    # Maximum maintenance interval
    # ----------------------------------------------------------

    for item_pos in range(number_items):

        for start in range(
            horizon - maximum_interval + 1
        ):

            row = np.zeros(number_variables)

            for period_pos in range(
                start,
                start + maximum_interval
            ):

                row[
                    index(item_pos, period_pos)
                ] = 1

            rows.append(row)

            lower_bounds.append(1)
            upper_bounds.append(np.inf)

    # ----------------------------------------------------------
    # Equipment age and probability constraints
    # ----------------------------------------------------------

    for item_pos, (_, item) in enumerate(
        equipment.iterrows()
    ):

        maximum_age = (
            (1 - item["beta"]) / item["alpha"]
        )

        for period_pos in range(horizon):

            row = np.zeros(number_variables)

            for maintenance_pos in range(
                period_pos + 1
            ):

                row[
                    index(item_pos, maintenance_pos)
                ] = (
                    recovery
                    * effectiveness_arr[
                        item_pos,
                        maintenance_pos
                    ]
                )

            age_without_maintenance = (
                initial_ages_arr[item_pos]
                + period_pos
                + 1
            )

            rows.append(row)

            lower_bounds.append(-np.inf)
            upper_bounds.append(
                age_without_maintenance
            )

            if limit_probability:

                rows.append(row.copy())

                lower_bounds.append(
                    age_without_maintenance
                    - maximum_age
                )

                upper_bounds.append(np.inf)

    constraint_matrix = sparse.csr_matrix(
        np.vstack(rows)
    )

    result = milp(
        c=objective_coefficients,
        integrality=np.ones(number_variables),
        bounds=Bounds(
            np.zeros(number_variables),
            np.ones(number_variables)
        ),
        constraints=LinearConstraint(
            constraint_matrix,
            np.asarray(lower_bounds),
            np.asarray(upper_bounds)
        ),
        options={
            "time_limit": 120
        }
    )

    if not result.success:

        return {
            "success": False,
            "message": result.message
        }

    decisions = np.rint(
        result.x
    ).astype(int).reshape(
        number_items,
        horizon
    )

    records = []

    total_objective = 0.0

    for item_pos, (_, item) in enumerate(
        equipment.iterrows()
    ):

        age = initial_ages_arr[item_pos]

        for period_pos in range(horizon):

            age = (
                age
                + 1
                - recovery
                * effectiveness_arr[
                    item_pos,
                    period_pos
                ]
                * decisions[
                    item_pos,
                    period_pos
                ]
            )

            age = max(0.0, age)

            probability = np.clip(
                item["alpha"] * age
                + item["beta"],
                0,
                1
            )

            cost = (
                probability
                * item["failure_cost"]
                / (period_pos + 1)
            )

            total_objective += cost

            records.append({
                "item": int(item["item"]),
                "period": period_pos + 1,
                "maintenance":
                    decisions[
                        item_pos,
                        period_pos
                    ],
                "relative_age": age,
                "failure_probability":
                    probability,
                "weighted_expected_cost":
                    cost,
                "pm_cost":
                    item["pm_cost"]
                    * decisions[
                        item_pos,
                        period_pos
                    ],
                "labor":
                    item["labor"]
                    * decisions[
                        item_pos,
                        period_pos
                    ],
            })

    schedule = pd.DataFrame(
        decisions,
        index=equipment["item"].astype(int),
        columns=np.arange(1, horizon + 1)
    )

    return {
        "success": True,
        "objective": total_objective,
        "schedule": schedule,
        "detail": pd.DataFrame(records),
        "solver": result
    }


# ============================================================
# Synthetic instance generation
# ============================================================

def generate_instance(
    name,
    number_items,
    horizon,
    criticality="mixed",
    seed=1,
):

    local_rng = np.random.default_rng(seed)

    lambdas = local_rng.uniform(
        0.05,
        0.18,
        number_items
    )

    parameters = [
        fit_exponential_curve(
            lambda_val,
            horizon
        )
        for lambda_val in lambdas
    ]

    if criticality == "high":

        failure_costs = local_rng.lognormal(
            mean=np.log(100),
            sigma=0.45,
            size=number_items
        )

    elif criticality == "low":

        failure_costs = local_rng.lognormal(
            mean=np.log(45),
            sigma=0.35,
            size=number_items
        )

    else:

        failure_costs = local_rng.lognormal(
            mean=np.log(70),
            sigma=0.50,
            size=number_items
        )

    pm_costs = np.maximum(
        2,
        np.round(
            failure_costs
            * local_rng.uniform(
                0.06,
                0.16,
                number_items
            )
        )
    )

    labor = local_rng.integers(
        2,
        9,
        number_items
    )

    equipment = pd.DataFrame({
        "item":
            np.arange(1, number_items + 1),

        "lambda":
            lambdas,

        "alpha":
            [value[0] for value in parameters],

        "beta":
            [value[1] for value in parameters],

        "failure_cost":
            np.round(failure_costs, 2),

        "pm_cost":
            pm_costs.astype(float),

        "labor":
            labor.astype(float),

        "linearization_error_pct":
            [value[2] for value in parameters]
    })

    budget = float(
        np.ceil(
            equipment["pm_cost"].sum()
            / 5
            * 1.45
            * 1.35
        )
    )

    labor_capacity = float(
        np.ceil(
            equipment["labor"].sum()
            / 5
            * 1.45
            * 1.35
        )
    )

    return {
        "name": name,
        "equipment": equipment,
        "horizon": horizon,
        "budget": budget,
        "labor_capacity": labor_capacity
    }


# ============================================================
# Monte Carlo simulation
# ============================================================

def monte_carlo_simulation(
    equipment,
    schedule,
    horizon,
    recovery,
    initial_age,
    mean_mu,
    std_mu,
    replications=5000,
    correlation=0.0,
    seed=1234,
):

    local_rng = np.random.default_rng(seed)

    number_items = len(equipment)

    decisions = schedule.to_numpy(
        dtype=float
    )

    maximum_std = (
        np.sqrt(
            mean_mu
            * (1 - mean_mu)
        )
        * 0.995
    )

    used_std = min(
        std_mu,
        maximum_std
    )

    concentration = (
        mean_mu
        * (1 - mean_mu)
        / used_std**2
        - 1
    )

    parameter_a = (
        mean_mu
        * concentration
    )

    parameter_b = (
        (1 - mean_mu)
        * concentration
    )

    independent_z = local_rng.normal(
        size=(
            replications,
            number_items,
            horizon
        )
    )

    if correlation > 0:

        common_z = local_rng.normal(
            size=(
                replications,
                1,
                horizon
            )
        )

        final_z = (
            np.sqrt(correlation)
            * common_z
            + np.sqrt(1 - correlation)
            * independent_z
        )

    else:

        final_z = independent_z

    uniforms = norm.cdf(final_z)

    effectiveness = beta_dist.ppf(
        np.clip(
            uniforms,
            1e-10,
            1 - 1e-10
        ),
        parameter_a,
        parameter_b
    )

    ages = np.zeros(
        (
            replications,
            number_items,
            horizon
        )
    )

    state = np.full(
        (
            replications,
            number_items
        ),
        initial_age,
        dtype=float
    )

    for period_pos in range(horizon):

        state = (
            state
            + 1
            - recovery
            * effectiveness[
                :,
                :,
                period_pos
            ]
            * decisions[
                :,
                period_pos
            ][None, :]
        )

        ages[
            :,
            :,
            period_pos
        ] = np.maximum(
            state,
            0
        )

    alpha_arr = (
        equipment["alpha"]
        .to_numpy()[None, :, None]
    )

    beta_arr = (
        equipment["beta"]
        .to_numpy()[None, :, None]
    )

    lambda_arr = (
        equipment["lambda"]
        .to_numpy()[None, :, None]
    )

    costs_arr = (
        equipment["failure_cost"]
        .to_numpy()[None, :, None]
    )

    weights = (
        1
        / np.arange(
            1,
            horizon + 1
        )
    )[None, None, :]

    linear_probability = np.clip(
        alpha_arr * ages
        + beta_arr,
        0,
        1
    )

    exponential_probability = (
        1
        - np.exp(
            -lambda_arr * ages
        )
    )

    linear_cost = np.sum(
        linear_probability
        * costs_arr
        * weights,
        axis=(1, 2)
    )

    exponential_cost = np.sum(
        exponential_probability
        * costs_arr
        * weights,
        axis=(1, 2)
    )

    return (
        linear_cost,
        exponential_cost
    )


# ============================================================
# Risk measures
# ============================================================

def risk_metrics(values):

    var95 = np.quantile(
        values,
        0.95
    )

    var99 = np.quantile(
        values,
        0.99
    )

    return {
        "mean":
            values.mean(),

        "std":
            values.std(ddof=1),

        "p05":
            np.quantile(values, 0.05),

        "median":
            np.median(values),

        "VaR95":
            var95,

        "CVaR95":
            values[
                values >= var95
            ].mean(),

        "VaR99":
            var99,

        "CVaR99":
            values[
                values >= var99
            ].mean()
    }


# ============================================================
# Main computational experiment
# ============================================================

def run_example(
    directory="model_results"
):

    folder = Path(directory)

    folder.mkdir(
        exist_ok=True
    )

    equipment = pd.DataFrame({
        "item":
            [1, 2, 3, 4, 5],

        "lambda":
            [0.10, 0.15, 0.12, 0.09, 0.08],

        "alpha":
            [0.0503, 0.0560, 0.0534, 0.0482, 0.0457],

        "beta":
            [0.0483, 0.1159, 0.0745, 0.0360, 0.0245],

        "failure_cost":
            [50, 40, 100, 60, 55],

        "pm_cost":
            [5, 7, 12, 8, 4],

        "labor":
            [4, 5, 10, 5, 6]
    })

    result = solve_model(
        equipment,
        horizon=15,
        budget=12,
        labor_capacity=10
    )

    if not result["success"]:

        raise RuntimeError(
            result["message"]
        )

    scenarios = [
        (
            "Optimistic",
            0.95,
            0.03,
            0.10
        ),
        (
            "Baseline",
            0.85,
            0.08,
            0.00
        ),
        (
            "High variance",
            0.85,
            0.18,
            0.00
        ),
        (
            "Correlated",
            0.85,
            0.12,
            0.60
        ),
        (
            "Degraded",
            0.65,
            0.10,
            0.30
        ),
        (
            "Severe",
            0.50,
            0.15,
            0.60
        )
    ]

    risk_rows = []

    for scenario_index, (
        name,
        mean,
        std,
        correlation
    ) in enumerate(
        tqdm(
            scenarios,
            desc="Monte Carlo"
        )
    ):

        linear, exponential = (
            monte_carlo_simulation(
                equipment,
                result["schedule"],
                horizon=15,
                recovery=4,
                initial_age=3,
                mean_mu=mean,
                std_mu=std,
                replications=5000,
                correlation=correlation,
                seed=2026 + scenario_index
            )
        )

        for model, values in [
            ("Linear", linear),
            ("Exponential", exponential)
        ]:

            risk_rows.append({
                "scenario":
                    name,

                "model":
                    model,

                "mean_mu":
                    mean,

                "std_mu":
                    std,

                "correlation":
                    correlation,

                **risk_metrics(values)
            })

    # --------------------------------------------------------
    # Synthetic instances
    # --------------------------------------------------------

    instances = [
        generate_instance(
            "Bottling line",
            6,
            15,
            seed=11
        ),

        generate_instance(
            "Pumping station",
            10,
            20,
            criticality="high",
            seed=22
        ),

        generate_instance(
            "Cold-storage facility",
            15,
            26,
            seed=33
        ),

        generate_instance(
            "Multi-product factory",
            25,
            30,
            seed=44
        )
    ]

    instance_rows = []

    solutions = {}

    for instance in tqdm(
        instances,
        desc="Instances"
    ):

        solution = solve_model(
            instance["equipment"],
            instance["horizon"],
            instance["budget"],
            instance["labor_capacity"]
        )

        solutions[
            instance["name"]
        ] = solution

        instance_rows.append({
            "instance":
                instance["name"],

            "items":
                len(
                    instance["equipment"]
                ),

            "horizon":
                instance["horizon"],

            "status":
                (
                    "Optimal"
                    if solution["success"]
                    else "Infeasible"
                ),

            "objective":
                solution.get(
                    "objective",
                    np.nan
                ),

            "interventions":
                (
                    solution["detail"][
                        "maintenance"
                    ].sum()
                    if solution["success"]
                    else np.nan
                )
        })

    # --------------------------------------------------------
    # Export results
    # --------------------------------------------------------

    excel_path = (
        folder
        / "maintenance_model_results.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="xlsxwriter"
    ) as writer:

        equipment.to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )

        result["schedule"].to_excel(
            writer,
            sheet_name="Schedule"
        )

        result["detail"].to_excel(
            writer,
            sheet_name="Details",
            index=False
        )

        pd.DataFrame(
            risk_rows
        ).to_excel(
            writer,
            sheet_name="Monte_Carlo",
            index=False
        )

        pd.DataFrame(
            instance_rows
        ).to_excel(
            writer,
            sheet_name="Instances",
            index=False
        )

        for position, instance in enumerate(
            instances,
            start=1
        ):

            instance["equipment"].to_excel(
                writer,
                sheet_name=f"Synthetic_data_{position}",
                index=False
            )

            if solutions[
                instance["name"]
            ]["success"]:

                solutions[
                    instance["name"]
                ]["schedule"].to_excel(
                    writer,
                    sheet_name=f"Synthetic_plan_{position}"
                )

    print(excel_path)


if __name__ == "__main__":
    run_example()
