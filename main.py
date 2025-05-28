from scripts.preprocess import clean_ufc_data
from scripts.preprocess import clean_ufc_data, build_fighter_profiles

# Run the clean function on the UFC data
df = clean_ufc_data("data/ufc-master.csv")

fighter_profiles = build_fighter_profiles(df)

print(fighter_profiles[fighter_profiles['Fighter'] == 'Jon Jones'])
