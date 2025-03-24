from ucimlrepo import fetch_ucirepo 
import pandas as pd 
from credit_mapping import attribute_mapping
import json

def load_dataset():
    statlog_german_credit_data = fetch_ucirepo(id=144)

    X = statlog_german_credit_data.data.features 
    y = statlog_german_credit_data.data.targets 
    
    df = pd.concat([X, y], axis=1)

    metadata = statlog_german_credit_data.variables

    return df, metadata

def get_nl_df(df):
    for column in df.columns:
        if column in attribute_mapping:
            df[column] = df[column].map(attribute_mapping[column]).fillna(df[column])

    return df

def generate_bulleted_profile(row, metadata):
    profile = []
    
    for col in row.index:
        if col in attribute_mapping:
            if col == 'class': continue
            value = row[col]
            if value in attribute_mapping[col]:
                description = attribute_mapping[col][value]
                profile.append(f"- {metadata.loc[metadata['name'] == col, 'description'].values[0]}: {description}")
            else:
                profile.append(f"- {metadata.loc[metadata['name'] == col, 'description'].values[0]}: {value}")
        else:
            profile.append(f"- {metadata.loc[metadata['name'] == col, 'description'].values[0]}: {row[col]}")
    
    return '\n'.join(profile)

def generate_natural_language_profile(row, metadata):
    profile = []
    profile.append("This individual has the following credit profile: \n")
    for col in row.index:
        if col == 'class': continue
        if col in attribute_mapping:
            value = row[col]
            if value in attribute_mapping[col]:
                description = attribute_mapping[col][value]
                profile.append(f"The {metadata.loc[metadata['name'] == col, 'description'].values[0]} is {description}.")
            else:
                profile.append(f"The {metadata.loc[metadata['name'] == col, 'description'].values[0]} is {value}.")
        else:
            profile.append(f"The {metadata.loc[metadata['name'] == col, 'description'].values[0]} is {row[col]}.")
    
    profile.append(f"\nBased on these characteristics, this individual has been classified as having {row['class']}.\n")

    return ' '.join(profile)


def generate_jsonl_entry(row, metadata, bulleted=True):
    user_content = "Here is an individual's credit profile. Based on the following attributes, what can you infer about this person?"

    if bulleted: profile = generate_bulleted_profile(row, metadata)
    else: profile = generate_natural_language_profile(row, metadata)

    assistant_content = f"{row['class']}."

    entry = {
        "messages": [
            {"role": "user", "content": f"{user_content}\n{profile}"},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {"status_sex": row['Attribute9'], "age": row['Attribute13'], "foreign_worker": row['Attribute20']}
    }
    return entry

def generate_jsonl_file(df, metadata, bulleted=True):
    tag = 'bulleted' if bulleted else 'natural_language'
    filename = f'datasets/eval/credit_data/{tag}_credit_profiles.jsonl'

    with open(filename, 'w') as f:
        for index, row in df.iterrows():
            entry = generate_jsonl_entry(row, metadata, bulleted)
            json.dump(entry, f)
            f.write('\n')

def main():
    df, metadata = load_dataset()
    df = get_nl_df(df)

    print(df.head())
    print(df.iloc[0, :])

    attribute_info = ""
    for name, description, units in zip(metadata['name'], metadata['description'], metadata['units']):
        attribute_info += f"{name}: {description}"
        if units: attribute_info += f" ({units})"
        attribute_info += '/n'
    
    print(generate_bulleted_profile(df.iloc[0, :], metadata))
    print(generate_natural_language_profile(df.iloc[0, :], metadata))

    generate_jsonl_file(df, metadata, bulleted=True)
    generate_jsonl_file(df, metadata, bulleted=False)
    
main()