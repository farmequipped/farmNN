import pandas as pd

def combine_datasets(disaster_file, aq_file, output_file):
    print("Loading datasets...")
    # Load the datasets
    disasters_df = pd.read_csv(disaster_file)
    aq_df = pd.read_csv(aq_file)

    print("Cleaning and preparing data...")
    # 1. Standardize the FIPS codes so they match in both datasets
    # FEMA dataset FIPS codes
    disasters_df['fipsStateCode'] = pd.to_numeric(disasters_df['fipsStateCode'], errors='coerce')
    disasters_df['fipsCountyCode'] = pd.to_numeric(disasters_df['fipsCountyCode'], errors='coerce')
    
    # Air Quality dataset FIPS codes
    aq_df['State FIPS Code'] = pd.to_numeric(aq_df['State FIPS Code'], errors='coerce')
    aq_df['County FIPS Code'] = pd.to_numeric(aq_df['County FIPS Code'], errors='coerce')

    # 2. Standardize the Dates
    # Convert Air Quality Date to a datetime object
    aq_df['Date'] = pd.to_datetime(aq_df['Date'], errors='coerce')
    
    # Convert FEMA Incident Begin Date to a datetime object (and remove the timezone part for matching)
    disasters_df['incidentBeginDate'] = pd.to_datetime(disasters_df['incidentBeginDate'], errors='coerce').dt.tz_localize(None).dt.normalize()

    print("Merging datasets...")
    # 3. Merge the datasets
    # We will do a 'left' merge to keep all air quality data and add disaster info if a disaster started in that county on that date.
    # You can change 'how' to 'inner' if you only want rows where both a disaster and AQ reading occurred.
    combined_df = pd.merge(
        aq_df,
        disasters_df,
        left_on=['State FIPS Code', 'County FIPS Code', 'Date'],
        right_on=['fipsStateCode', 'fipsCountyCode', 'incidentBeginDate'],
        how='left'
    )

    print("Saving to CSV...")
    # 4. Save the combined data to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully saved combined data to {output_file}")

# Run the function
if __name__ == "__main__":
    disaster_csv = 'DisasterDeclarationsSummaries.csv'
    aq_csv = 'ad_viz_plotval_data (1).csv'
    output_csv = 'Combined_AQ_Disasters.csv'
    
    combine_datasets(disaster_csv, aq_csv, output_csv)