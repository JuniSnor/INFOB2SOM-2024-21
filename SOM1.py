# Import pandas to read and manipulate the dataset, google.generativeai for using the Gemini API, time for pauses and Counter to count each mechanic and its accuracy scores
import pandas as pd
import google.generativeai as genai
import time
from collections import Counter

# The class for analyzing the board game mechanics
class BoardGameMechanicsAnalyzer:
    def __init__(self, dataset_path, gemini_api_key):
        # Dataset and API key are not hardcoded into functions so they can be adjusted if needed and because it was required by the assignment to include them in the class
        self.dataset_path = dataset_path
        self.gemini_api_key = gemini_api_key 
        self.data = self.clean_data() # Call the clean_data function upon initializing the class so the data is clean for analysis
        genai.configure(api_key=self.gemini_api_key) # Configure the API key for the Gemini API
        self.model = genai.GenerativeModel("gemini-1.5-flash") # Choose the model to use
        self.mechanics_counter = Counter() # Initialize the mechanics counter
        self.analyzed_mechanics = set() # Initialize the set of analyzed mechanics

    # Read data and remove columns where name, year of publication, and mechanics are empty
    def clean_data(self):
        data = pd.read_csv(self.dataset_path, delimiter=';') # No idea why the delimiter is a semicolon, but this works now
        data = data.dropna(subset=['Name', 'Year Published', 'Mechanics']) # Remove rows where the columns Name, Year Published, or Mechanics are empty
        data = data.reset_index(drop=True) # Reset the row indices after removing invalid data
        data['Rating Average'] = data['Rating Average'].str.replace(',', '.').astype(float) # For the top 200, convert the rating average to a float because it currently uses commas
        return data

    # Verify the mechanics of a given game, main function of the class
    def verify_mechanics(self, game_name, year_published=None):

        if year_published:
            game_data = self.data[(self.data['Name'] == game_name) & (self.data['Year Published'] == year_published)]
        else:
            game_data = self.data[self.data['Name'] == game_name]
    
        if game_data.empty:
            return f"Game '{game_name}' not found in the dataset."
            
        mechanics_bgg = self.data.loc[self.data['Name'] == game_name, 'Mechanics'].values[0] # Look up the mechanics of the game in the dataset
        year_published = int(game_data['Year Published'].values[0]) # Look up the publication year of the game in the dataset

        response = self.model.generate_content(f"The game '{game_name}' has the following mechanics listed: {mechanics_bgg}. Is this correct? If not, what are the correct mechanics? Only list the mechanics that you think apply, separated by commas. Don't say anything else besides the mechanics you think apply").text
        
        similarity = self.calculate_similarity(mechanics_bgg, response) # Calculate the accuracy of the mechanics

        self.analyzed_mechanics.update(mechanics_bgg.split(', ')) # Update the analyzed_mechanics with the mechanics of the game, for consistency analysis
        return f"{game_name}, {year_published} has received an accuracy score of: {similarity['accuracy']:.3f}"

    # Convert BGG and Gemini mechanics strings to sets and calculate accuracy, return it as a dictionary
    def calculate_similarity(self, mechanics_bgg, response):
        bgg_mechanics = set(mechanics_bgg.replace('\n', '').split(', '))
        gemini_mechanics = set(response.replace('\n', '').split(', '))
        self.mechanics_counter.update(gemini_mechanics) # Update the counter with Gemini mechanics
        correct_predictions = bgg_mechanics.intersection(gemini_mechanics) # Calculate accuracy by comparing sets using intersection
        accuracy = len(correct_predictions) / len(gemini_mechanics) if bool(gemini_mechanics) else 0 # Calculate accuracy if Gemini mechanics contain at least one correct mechanic; otherwise, return 0
        
        return {
            'accuracy': accuracy,
            'bgg_mechanics': bgg_mechanics,
            'gemini_mechanics': gemini_mechanics,
            'correct_predictions': correct_predictions
        }
    # Verify the mechanics of multiple games, using a game list as input and passing each game to the verify_mechanics function then returning the results
    def verify_multiple_games(self, game_list):
        results = []
        for game in game_list:
            if isinstance(game, list):
                game_name = game[0]
            elif isinstance(game, tuple):
                game_name, year_published = game
                result = self.verify_mechanics(game_name, year_published)
            else:
                game_name = game
            result = self.verify_mechanics(game_name)
            results.append(result)
            time.sleep(4)  # Pause for 4 seconds between requests to avoid overloading the API
        return results

    # Verify all games in the dataset
    def verify_all_games(self): # Verify all games in the dataset
        game_list = self.data[['Name']].values.tolist() # Create a list of all games in the dataset
        return self.verify_multiple_games(game_list) # Verify the list

    # Calculate the mean accuracy of the top 200 games
    def mean_accuracy_top_200(self):
        top_200_games = self.data.nlargest(200, 'Rating Average')[['Name']].values.tolist()
        results = self.verify_multiple_games(top_200_games) # Verify the mechanics of the top 200 games
        accuracies = []
        for result in results:
            if 'accuracy' in result:
                accuracy_str = result.split('accuracy score of: ')[1]
                accuracies.append(float(accuracy_str))
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0 # Calculate the average accuracy of the top 200 games
        return mean_accuracy

    def analyze_mechanics_consistency(self):
        # Calculate the frequency of each mechanic assigned by Gemini
        most_common_mechanics = self.mechanics_counter.most_common(10)
        least_common_mechanics = self.mechanics_counter.most_common()[:-11:-1]
        all_mechanics = set(self.analyzed_mechanics) # Get all mechanics from the analyzed games
        assigned_mechanics = set(self.mechanics_counter.keys()) # Get the keys of the mechanics_counter to retrieve mechanics assigned by Gemini
        never_assigned_mechanics = all_mechanics - assigned_mechanics # Check which mechanics were never assigned by Gemini
        return most_common_mechanics, least_common_mechanics, never_assigned_mechanics

# WARNING: this function will take a long time to run, as it verifies all games in the dataset. It is recommended to run only the functions that analyse a specific game or a list of games.
# Example usage:
# change dataset_path to the path of the dataset on your machine and gemini_api_key to your Gemini API key.
dataset_path = r'C:\Users\Junior\Downloads\bgg_dataset.csv'
gemini_api_key = 'AIzaSyD7myuik42J6DvR2Tr1iGsgPd2WntZfxF0'
analyzer = BoardGameMechanicsAnalyzer(dataset_path, gemini_api_key)

# Calculate the accuracy of the mechanics for the game Gloomhaven
accuracy_game = analyzer.verify_mechanics('Gloomhaven', 2017)
print(f"Analyzed game: {accuracy_game}")

# Calculate the accuracy of the mechanics for the games Nemesis, Spirit Island, Wingspan, and Gloomhaven with a year to show it works
game_list = [('Nemesis'), ('Spirit Island'), ('Wingspan'), ('Gloomhaven', 2017)]
accuracy_multiple_games = analyzer.verify_multiple_games(game_list)
print(accuracy_multiple_games)

# Example to verify all games
accuracy_all_games = analyzer.verify_all_games()
print(accuracy_all_games)

# Example to calculate the mean accuracy for the top 200 games
mean_accuracy = analyzer.mean_accuracy_top_200()
print(f"Mean accuracy for the top 200 games: {mean_accuracy}")

# Example to analyze mechanics consistency
most_common, least_common, never_assigned = analyzer.analyze_mechanics_consistency()
print(f"Most common mechanics: {most_common}")
print(f"Least common mechanics: {least_common}")
print(f"Never assigned mechanics: {never_assigned}")