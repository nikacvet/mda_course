import numpy as np
import pandas as pd
import pycountry


def project_with_realistic_dynamics(df, targets, shocks, improvement_rate=0.95,
    shock_impact_range=(0.02, 0.06), seed=123, n_years=26, start_year=2025):

    np.random.seed(seed)
    years = list(range(start_year, start_year + n_years))
    results = {}

    for low_country, top_country in targets.items():
        low_row = df[df["country"] == low_country].iloc[0]
        top_row = df[df["country"] == top_country].iloc[0]

        low_score = low_row["Weighted_Score"]
        top_score = top_row["Weighted_Score"]
        target_score = low_score + improvement_rate * (top_score - low_score)

        current = [low_score]
        improved = [low_score]

        for i in range(1, n_years):
            year = years[i]

            # current country simulation with gradual decline/stasis and noise
            delta_cur = np.random.normal(loc=0, scale=0.01)
            current_score = np.clip(current[-1] + delta_cur, 0, 1)
            current.append(current_score)

            # improved score projection
            progress = i / (n_years - 1)
            target_progress = low_score + progress * (target_score - low_score)
            noise = np.random.normal(loc=0, scale=0.01 * (1 - progress))
            
            # apply shocks if they occur in this year
            if year in shocks:
                noise -= np.random.uniform(*shock_impact_range)

            improved_score = np.clip(target_progress + noise, 0, 1)
            improved.append(improved_score)

        country_name = pycountry.countries.get(alpha_2=low_country).name
        top_country_name = pycountry.countries.get(alpha_2=top_country).name
        explanation = (
            f"{country_name} is compared to {top_country_name}, a country with a comparable economic footprint "
            f"(GDP) but stronger EU participation metrics. The simulation projects how {country_name}'s performance "
            f"would evolve if it progressively adopted {int(improvement_rate*100)}% of {top_country_name}'s metrics. "
        )

        results[low_country] = {
            "years": years,
            "score_current": current,
            "score_improved": improved,
            "top_country": top_country_name,
            "explanation": explanation,
            # "fig": fig  # Not needed in Kedro output
        }

    # Remove plot objects for JSON
    json_output = {
        country: {
            "years": vals["years"],
            "score_current": vals["score_current"],
            "score_improved": vals["score_improved"],
            "top_country": vals["top_country"],
            "explanation": vals["explanation"]
        } for country, vals in results.items()
    }

    return json_output
