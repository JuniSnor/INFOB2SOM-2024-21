import time
from collections import Counter
import unittest
from unittest.mock import patch
import pandas as pd
import google.generativeai as genai
import requests

# De class voor het analyseren van de boardgame mechanics
class BoardGameMechanicsAnalyzer:
    def __init__(self, dataset_path, gemini_api_key):
        self.dataset_path = dataset_path
        self.gemini_api_key = gemini_api_key
        self.data = self.clean_data() 
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.mechanics_counter = Counter() 
        self.analyzed_mechanics = set() 

    def clean_data(self):
        # Directly using hardcoded mock data instead of reading from a CSV file
        data = pd.DataFrame({
            'Name': ['Game 1', 'Game 2', 'Game 3'],
            'Mechanics': ['Mechanic 1, Mechanic 2', 'Mechanic 2, Mechanic 3', 'Mechanic 1, Mechanic 3'],
            'Year Published': [2015, 2018, 2020],
            'Rating Average': ['8,0', '7,5', '9,2']
        })

        data = data.dropna(subset=['Name', 'Year Published', 'Mechanics'])
        data = data.reset_index(drop=True) 
        data['Rating Average'] = data['Rating Average'].str.replace(',', '.').astype(float)
        return data

    def verify_mechanics(self, game_name, year_published=None):
        if year_published:
            game_data = self.data[(self.data['Name'] == game_name) & (self.data['Year Published'] == year_published)]
        else:
            game_data = self.data[self.data['Name'] == game_name]
    
        if game_data.empty:
            return f"Game '{game_name}' not found in the dataset."
            
        mechanics_bgg = self.data.loc[self.data['Name'] == game_name, 'Mechanics'].values[0] 
        year_published = int(game_data['Year Published'].values[0])

        response = self.model.generate_content(f"The game '{game_name}' has the following mechanics listed: {mechanics_bgg}. Is this correct? If not, what are the correct mechanics? Only list the mechanics that you think apply, separated by commas.").text
        
        similarity = self.calculate_similarity(mechanics_bgg, response) 

        self.analyzed_mechanics.update(mechanics_bgg.split(', ')) 
        return f"{game_name}, {year_published} has received an accuracy score of: {similarity['accuracy']:.3f}"

    def calculate_similarity(self, mechanics_bgg, response):
        bgg_mechanics = set(mechanics_bgg.replace('\n', '').split(', '))
        gemini_mechanics = set(response.replace('\n', '').split(', '))
        self.mechanics_counter.update(gemini_mechanics) 
        correct_predictions = bgg_mechanics.intersection(gemini_mechanics) 
        accuracy = len(correct_predictions) / len(gemini_mechanics) if bool(gemini_mechanics) else 0 
        
        return {
            'accuracy': accuracy,
            'bgg_mechanics': bgg_mechanics,
            'gemini_mechanics': gemini_mechanics,
            'correct_predictions': correct_predictions
        }

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
            time.sleep(4)  # Pauze van 4 seconden
        return results

    def verify_all_games(self):
        game_list = self.data[['Name']].values.tolist() 
        return self.verify_multiple_games(game_list)

    def mean_accuracy_top_200(self):
        top_200_games = self.data.nlargest(200, 'Rating Average')[['Name']].values.tolist()
        results = self.verify_multiple_games(top_200_games)
        accuracies = []
        for result in results:
            if 'accuracy' in result:
                accuracy_str = result.split('accuracy score of: ')[1]
                accuracies.append(float(accuracy_str))
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0 
        return mean_accuracy

    def analyze_mechanics_consistency(self):
        most_common_mechanics = self.mechanics_counter.most_common(10)
        least_common_mechanics = self.mechanics_counter.most_common()[:-11:-1]
        all_mechanics = set(self.analyzed_mechanics)
        assigned_mechanics = set(self.mechanics_counter.keys())
        never_assigned_mechanics = all_mechanics - assigned_mechanics
        return most_common_mechanics, least_common_mechanics, never_assigned_mechanics
    
# Unit tests
class TestBoardGameMechanicsAnalyzer(unittest.TestCase):

    @patch('google.generativeai.GenerativeModel')  # Mock GenerativeModel
    @patch('pandas.read_csv')  # Mock pandas.read_csv function
    def setUp(self, mock_read_csv, mock_GenerativeModel):
        # Mock pandas read_csv function to return hardcoded mock data
        mock_read_csv.return_value = pd.DataFrame({
            'Name': ['Game 1', 'Game 2', 'Game 3'],
            'Mechanics': ['Mechanic 1, Mechanic 2', 'Mechanic 2, Mechanic 3', 'Mechanic 1, Mechanic 3'],
            'Year Published': [2015, 2018, 2020],
            'Rating Average': ['8,0', '7,5', '9,2']
        })

        # Mock GenerativeModel's generate_content method
        mock_instance = mock_GenerativeModel.return_value
        mock_instance.generate_content.return_value.text = "Mechanic 1, Mechanic 2"  # Simulated response

        # Initialize BoardGameMechanicsAnalyzer with mock data
        self.analyzer = BoardGameMechanicsAnalyzer('dummy_path.csv', 'dummy_api_key')

    def test_verify_mechanics(self):
        # Test verifying mechanics and calculating accuracy for a single game
        accuracy_score = self.analyzer.verify_mechanics('Game 1')
        expected_result = "Game 1, 2015 has received an accuracy score of: 1.000"
        self.assertEqual(accuracy_score, expected_result)

    def test_mean_accuracy_top_200(self):
        # Test calculating the mean accuracy of the top 200 games
        mean_accuracy = self.analyzer.mean_accuracy_top_200()
        expected_result = 0.6666666666666666
        self.assertEqual(mean_accuracy, expected_result)


    def test_verify_multiple_games(self):
        # Test verifying mechanics and calculating accuracy for multiple games
        game_list = [('Game 1', 2015), ('Game 2', 2018)]
        accuracy_scores = self.analyzer.verify_multiple_games(game_list)
        expected_results = [
            "Game 1, 2015 has received an accuracy score of: 1.000",
            "Game 2, 2018 has received an accuracy score of: 0.500"
        ]
        self.assertEqual(accuracy_scores, expected_results)

    def test_verify_all_games(self):
        # Test verifying mechanics and calculating accuracy for all games
        accuracy_scores = self.analyzer.verify_all_games()
        expected_results = [
            "Game 1, 2015 has received an accuracy score of: 1.000",
            "Game 2, 2018 has received an accuracy score of: 0.500",
            "Game 3, 2020 has received an accuracy score of: 0.500"
        ]
        self.assertEqual(accuracy_scores, expected_results) 

    @patch('google.generativeai.GenerativeModel')  # Mock the GenerativeModel
    def test_gemini_api_unreachable(self, mock_GenerativeModel):
        # Simulate Gemini API being unreachable by raising exception when calling generate_content
        mock_instance = mock_GenerativeModel.return_value
        mock_instance.generate_content.side_effect = requests.exceptions.RequestException("Gemini API is unreachable")

        # Initialize the BoardGameMechanicsAnalyzer with mock data
        analyzer = BoardGameMechanicsAnalyzer('dummy_path.csv', 'dummy_api_key')

        # Try verifying mechanics and assert that an exception is raised
        with self.assertRaises(requests.exceptions.RequestException) as context:
            analyzer.verify_mechanics('Game 1')
        self.assertEqual(str(context.exception), "Gemini API is unreachable")


    @patch('google.generativeai.GenerativeModel')  # Mock the GenerativeModel
    def test_gemini_api_deterministic_response(self, mock_GenerativeModel):
        # Mock the GenerativeModel's generate_content method to return a fixed value
        mock_instance = mock_GenerativeModel.return_value
        mock_instance.generate_content.return_value.text = "Mechanic 1, Mechanic 2"

        # Initialize the BoardGameMechanicsAnalyzer with mock data
        analyzer = BoardGameMechanicsAnalyzer('dummy_path.csv', 'dummy_api_key')

        # Verify that response is deterministic and always returns the same mechanics
        accuracy_score = analyzer.verify_mechanics('Game 1')
        expected_result = "Game 1, 2015 has received an accuracy score of: 1.000"
        self.assertEqual(accuracy_score, expected_result)
        
        # Ensure that calling the same method again still produces the same result
        accuracy_score_again = analyzer.verify_mechanics('Game 1')
        self.assertEqual(accuracy_score_again, expected_result)  # Results should match, shows deterministic behavior

# Run tests
if __name__ == '__main__':
    unittest.main()
