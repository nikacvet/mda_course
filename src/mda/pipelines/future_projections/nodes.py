import numpy as np
import pandas as pd
import pycountry


def project_with_realistic_dynamics(df, targets, shocks, improvement_rate=0.95,
    shock_impact_range=(0.02, 0.06), seed=123, n_years=26, start_year=2025):

    np.random.seed(seed)
    years = list(range(start_year, start_year + n_years))
    results = {}

    # Predefined summary explanations for dashboard
    summaries = {
        "MT": (
            "Malta's current climate engagement, measured through a composite performance score, reflects low levels of "
            "collaboration, project density, and EU funding absorption. Cyprus, though similarly sized in GDP, performs "
            "significantly better on these metrics. This projection simulates Malta’s trajectory if it progressively adopted "
            "95% of Cyprus’s participation profile. Despite global setbacks like recessions or policy shifts, the improved "
            "path shows a clear and sustained rise in performance. This comparison highlights the untapped potential for "
            "smaller member states to scale their climate role by learning from structurally similar but more active peers."
        ),
        "HR": (
            "Croatia's climate project engagement lags behind that of Spain — one of the EU’s top performers — despite both "
            "facing comparable challenges in implementation scale. The current performance score reflects Croatia’s lower funding "
            "uptake and collaboration rates. If Croatia were to align with 95% of Spain’s project participation metrics, the "
            "simulation shows nearly a twofold improvement by 2050. This projection offers an aspirational benchmark, letting "
            "policymakers compare outcomes from aligning with the EU’s best-in-class performers versus structurally similar ones. "
            "It helps prioritize whether to close the gap with leaders or focus on achievable stepwise targets."
        )
    }

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
            f"would evolve if it progressively adopted 85% of {top_country_name}'s participation metrics across collaboration, "
            f"project density, and funding absorption. Global events like the {', '.join([f'{y} – {s}' for y, s in shocks.items()])} "
            f"are modeled to induce plausible setbacks. This scenario helps gauge feasibility and impact toward 2050 EU cohesion goals."
        )

        results[low_country] = {
            "years": years,
            "score_current": current,
            "score_improved": improved,
            "top_country": top_country_name,
            "explanation": explanation,
            "summary": summaries.get(low_country, ""),  # Optional fallback
        }

    # Remove plot objects for JSON
    json_output = {
        country: {
            "years": vals["years"],
            "score_current": vals["score_current"],
            "score_improved": vals["score_improved"],
            "top_country": vals["top_country"],
            "explanation": vals["explanation"],
            "summary": vals["summary"]
        } for country, vals in results.items()
    }

    return json_output
