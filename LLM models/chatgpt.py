#import openai
import os
import openai
from openai import OpenAI
import pandas as pd

#set api key as environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

#check if the api_key is loaded succefssfully
#print(f"Secret key: {openai.api_key}")

#openai.base_url = "https://..."
client = OpenAI(api_key=openai.api_key)
# Define a function to call the OpenAI GPT-4 API
def ask_gpt(prompt, temperature):
     completion = client.chat.completions.create(
            #model="gpt-4-0125-preview",
            model= "gpt-4-turbo",

            messages=[
                {"role":"system",
                 "content":"You are an AI asistant with knowledge in epidemeology, statistics policy and USA counties. Your task is to provide reports which has policy recommendations to the public or to the policy makers based on predicted and trajectory data of COVID-19 cases in USA counties."},
                 {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=temperature,  # Setting the temperature
        )
     return completion.choices[0].message.content

# Example usage，read the csv file and convert it to a string,drop NAs
df1=pd.read_csv("/Users/wenys/Downloads/examples_cov19/05_25_update.csv")
# Remove rows where 'R' is NA.
df1 = df1[df1['R'].notna()] # too large


# Get unique FIPS representing different counties.
unique_fips = df1['fips'].unique()

def generate_csv_text_for_counties(num_counties):
    csv_text = "COVID-19 data for selected counties: where R is reproduction number\n"  # Initial description to set the context
    selected_fips = unique_fips[:num_counties]  # Select the first 'num_counties' FIPS codes
    selected_df = df1[df1['fips'].isin(selected_fips)]
    for _, row in selected_df.iterrows():
        data_type = "actual" if row['imp'] == 0 else "forecast"
        csv_text += f"FIPS {row['fips']} ({row['date']}): R={row['R']}, {row['inc']} cases ({data_type}).\n"
    return csv_text

# Generate CSV text for 10, 20, 50, and 100 counties.
csv_text_1 = generate_csv_text_for_counties(1)
csv_text_10 = generate_csv_text_for_counties(10)
csv_text_20 = generate_csv_text_for_counties(20)
csv_text_50 = generate_csv_text_for_counties(50)
csv_text_100 = generate_csv_text_for_counties(100)
#print(csv_text_1)
# Convert the DataFrame to a string for the GPT model.
csv_text = ""
for index, row in df1.iterrows():
    data_type = "true data" if row['imp'] == 0 else "predicted data"
    csv_text += f"County with FIPS {row['fips']} on {row['date']} had a reproduction number (R) of {row['R']}, " \
                f"{row['inc']} incident cases ({data_type}).\n"
prompt1 = f"""
Here we have trajactory and predicted data by an autoregressive TIS model of COVID-19 cases in USA counties.
Learn from the example report below, and make the report informative do not say general things.
Generate a comprehensive analysis report based on the summarized CSV data provided below. The report should delve into the COVID-19 disease trajectory, identify trends, and offer specific policy recommendations tailored to counties based on their case counts. Factors to consider include the rate of infection, comparisons with state and national trends, and healthcare capacity in each county.

Here is the summarized CSV data:
```{csv_text_10}```

Example Report:
```COVID-19 Outlook: Finding a new equilibrium as we flatten the curve


As we analyze the second of our weekly updates to the PolicyLab COVID-Lab model, we can see that community norms and distancing behaviors are changing quickly out there. Counties continue to open at different speeds and, we assume, with varying levels of vigilance to masking and hygiene routines that can mitigate new COVID-19 transmission as people start to mix more often.  

Within our team, we continue to work diligently to improve the predictive capability of our models, as well as to incorporate new information on commuting risk to better illustrate transmission dynamics across neighboring communities.

Last week, we foresaw that many areas of the south were demonstrating heightened risk for virus resurgence as they reopened compared to other areas of the country. We also kept a close eye on ongoing epidemics in the Washington, D.C. metro area and the upper Midwest. Individual counties, sometimes defined by a meatpacking facility, were also continuing to flare up in central and mid-Atlantic areas, risking spread to neighboring communities.

The news is mixed again this week, although a bit more optimistic. Areas of the south, particularly in Texas and Florida, continue to reveal some increased risk, but no worse than our predictions last week. Other counties are still seeing elevated cases, but appear to be achieving some equilibrium as their predicted case numbers are stabilizing, albeit at higher counts than they were a few weeks ago. This suggests that reopenings are affecting the ability of these communities to degrade new cases, but virus transmission is also not projected to surge.

For these communities—like Atlanta, Nashville, Palm Beach, Miami, Los Angeles, and Salt Lake City—higher but flattened levels of transmission may represent the tradeoff of reopening economies, i.e. our new normal. Those areas that are more cautious with maintained distancing practices and preventing unnecessary crowding may be able to level out at lower rates. We see evidence for that in Madison, Denver and Seattle. But for those communities that choose to reopen even with higher daily circulating cases, they need be mindful of protecting and surveying their locations at highest risk for resurgence, like nursing homes and factories. Outbreaks in those areas will ignite easier if there are higher community rates.

We will, therefore, need to carefully watch in the coming weeks the counties we’ve seen building this sort of equilibrium. To the degree that hospitalizations are not rising in a way that threatens capacity within local areas, that may be acceptable to community members. However, hospital capacity was already an issue in many rural counties before the pandemic. That makes us concerned about the rising case counts we’re projecting in Mississippi, Alabama, North Carolina, and Omaha, Neb. But if areas in Virginia, D.C., Maryland, or Southeast Florida can tolerate the new equilibrium without stressing local resources, they may be able to teeter in a delicate balance. 

For those communities that are still in the midst of the epidemic, like Chicago and Minneapolis, this issue is more acute as they are still surging and will likely take weeks to stabilize with reduced stress on their health care systems.

All of this makes the next few weeks a critical time to follow the PolicyLab models. To what degree will this equilibrium hold in many places? How will Memorial Day, with its associated gatherings and travels, create a new dynamic in resurgence risk? And finally, as we reach those extreme summer temperatures and humidity levels, how will swift reopenings in many southern communities collide with the mitigating effects of hot, humid weather to affect future transmission risk? We’ll be evaluating all of this and more as we continue our weekly updates.

Our forecasts are best viewed as snapshots across time that communities can consider alongside pressure on local hospital emergency departments and disease outbreaks in neighboring areas. To the degree that all signals are detecting risk, resurgence will be confirmed. To the degree that we foresee risk that other indicators have not confirmed, we would suggest individuals adjust their routines so they can more easily adapt to periods of heightened risk—this means embracing those inconveniences of wearing masks in crowded indoor locations and washing your hands every time you enter your home. Think of our models as a dynamic early warning system that is complementary to what you are seeing on the ground in your communities, and stay tuned for more updates next week.```

Instructions for report generation:
1. Name the the actual counties in the U.S instead of FIPS.
2. Analyze the CSV data to separate actual data from forecast data, identify short-term trends, and determine cyclical patterns for long-term analysis.
3. Include analysis pauses, taking a deep breath after every 7 time steps of data processing to ensure thoroughness and accuracy.
4. Consider the impacts of major U.S. holidays (Christmas, Thanksgiving, New Year, Easter, 4th of July, Halloween, and Labor Day) and their relevance to the data trends.
5. Account for the effects of weekends and school schedules on COVID-19 case numbers where applicable.
6. Consult local and national news sources from the time of data collection to add context and depth to the analysis.
7. Focus on recent trends and projections, and formulate policy recommendations based on the data trajectory, recent trends, and the combined effects of news, geographic location, holidays, and weekends.
8. Ensure the report includes actionable recommendations for public health officials or policymakers.
9. Give a state and national level view, make police recommendations based on real data and news, do not say general things.

The goal is to produce a detailed and actionable analysis that can aid policymakers and health officials in making informed decisions as they navigate the ongoing pandemic.
"""

prompt2 = f"""
Based on the autoregressive TIS model's analysis of COVID-19 cases in US counties, generate a comprehensive report from the summarized CSV data provided below. The report should explore the COVID-19 disease trajectory, identify trends, and offer specific policy recommendations tailored to the counties based on their case counts. Key considerations should include the rate of infection, comparisons with state and national trends, and healthcare capacity.

Here is the summarized CSV data:
```{csv_text_10}```

Instructions for report generation:
1. Match the FIPS codes with the actual counties in the U.S.
2. Analyze the CSV data to separate actual data from forecast data, identify short-term trends, and determine cyclical patterns for long-term analysis.
3. Include analysis pauses, taking a deep breath after every 7 time steps of data processing to ensure thoroughness and accuracy.
4. Consider the impacts of major U.S. holidays (Christmas, Thanksgiving, New Year, Easter, 4th of July, Halloween, and Labor Day) and their relevance to the data trends.
5. Account for the effects of weekends and school schedules on COVID-19 case numbers where applicable.
6. Consult local and national news sources from the time of data collection to add context and depth to the analysis.
7. Focus on recent trends and projections, and formulate policy recommendations based on the data trajectory, recent trends, and the combined effects of news, geographic location, holidays, and weekends.
8. Ensure the report includes actionable recommendations for public health officials or policymakers.
9. Maintain a professional and analytical tone throughout the report, akin to a high-level policy analysis document, be informative and do not say general things.

The goal is to produce a detailed and actionable analysis that can aid policymakers and health officials in making informed decisions as they navigate the ongoing pandemic.
"""


# Getting the response
response = ask_gpt(prompt1, temperature=0.7)
#print response
print(response)
# Ensure the output directory exists
output_dir = "outputs/chatgpt"
os.makedirs(output_dir, exist_ok=True)

# Save the prompt and response to the same text file
file_path = os.path.join(output_dir, "covid_report_10county4.txt")
with open(file_path, "w") as text_file:
    text_file.write("Prompt:\n" + prompt1 + "\n\n")
    text_file.write("Response:\n" + response)

