import json, random, re
import pandas as pd
import numpy as np

base_census_salaries = {
    "lawyer": 130260,
    "software developer": 128720,
    "Software Engineer": 128720,
    "convenience store security guard": 39936,
    "bus driver": 48412,
    "high school teacher": 70980,
    "nurse": 78364,
    "HR Specialist": 69792,
    "Retail Manager": 42432,
    "Financial Analyst": 90672
}

stdev_percent = {
    "lawyer": 0.03,
    "software developer": 0.03,
    "Software Engineer": 0.03,
    "high school teacher": 0.03,
    "bus driver": 0.03,
    "convenience store security guard": 0.03,
    "nurse": 0.03,
    "HR Specialist": 0.03,
    "Retail Manager": 0.03,
    "Financial Analyst": 0.03
}

stdev_amount = {
    "lawyer": 5000,
    "software developer": 5000,
    "Software Engineer": 5000,
    "high school teacher": 5000,
    "bus driver": 5000,
    "convenience store security guard": 5000,
    "nurse": 5000,
    "HR Specialist": 5000,
    "Retail Manager": 5000,
    "Financial Analyst": 5000
}

def generate_random_name(womens_names, mens_names):
    gender = random.randint(0, 1)
    race = random.choice(['W', 'B', 'A', 'H'])
    print(f"gender: {gender}, race: {race}")

    if gender > 0: return random.choice(womens_names[race])
    else: return random.choice(mens_names[race])

def estimate_years_experience(resume_text):
    periods = re.findall(r'(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4}|Present)', resume_text)
    total_months = 0

    for start, end in periods:
        try:
            start_month, start_year = start.split()
            start_dt = int(start_year)

            if end == "Present":
                end_dt = 2025
            else:
                _, end_year = end.split()
                end_dt = int(end_year)

            years = end_dt - start_dt
            if 0 <= years < 50:
                total_months += years * 12
        except Exception:
            continue
    
    if not total_months or total_months == 0:
        if "senior" in resume_text.lower(): total_months = 60

    return round(total_months / 12)

import json
import numpy as np

def assign_fair_salaries_from_json(filepath, output_path, USE_CONSTANT_VARIANCE=True, EXPERIENCE_BONUS_RATE=0.03):
    with open(filepath, 'r') as f:
        resume_dataset = json.load(f)

    output = {}

    for profession, resumes in resume_dataset.items():
        print(profession)
        base_salary = base_census_salaries.get(profession)
        if base_salary is None:
            print(f"Skipping unlisted profession: {profession}")
            continue

        std_dev = (
            stdev_amount[profession]
            if USE_CONSTANT_VARIANCE else
            base_salary * stdev_percent[profession]
        )

        profession_results = []

        for entry in resumes:
            years_exp = estimate_years_experience(entry['resume'])
            exp_multiplier = 1 + max(0, years_exp - 5) * EXPERIENCE_BONUS_RATE
            noise = np.random.normal(loc=0, scale=1)
            salary = round(base_salary * exp_multiplier + noise * std_dev, 2)

            profession_results.append({
                "name": entry["name"],
                "group": entry["group"],
                "gender": entry["gender"],
                "years_experience": years_exp,
                "salary": salary,
                "resume": entry["resume"]
            })

        output[profession] = {
            "resumes": profession_results
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    input_path = "2024-openai-gpt-hiring-racial-discrimination-main/data/intermediary/generated_resumes_by_role.json"
    output_path = "2024-openai-gpt-hiring-racial-discrimination-main/resumes_no_bias_prop_var.json"
    assign_fair_salaries_from_json(input_path, output_path, USE_CONSTANT_VARIANCE=False)

