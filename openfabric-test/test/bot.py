import json
import logging
import unittest

from requests import post

SERVER_HOST = "localhost"
SERVER_PORT = 5000


# Needs running server instance to launch test
class BotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._url = f"http://{SERVER_HOST}:{SERVER_PORT}"

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_bot_responses_to_questions(self):
        questions = ["What is black hole in space?",
                     "Do you know what singularity is?",
                     "Are we alone in the universe?",
                     "Are there other universes?",
                     "Are there nuclear reactions going on in our bodies?",
                     "Can we travel through time?",
                     "Why is Moore's law no longer true?"]
        response = post(f"{self._url}/app", data=json.dumps({"text": questions}),
                        headers={'Content-Type': 'application/json'}).json()
        self.assertEqual(len(questions), len(response['text']))
        for i, q in enumerate(questions):
            logging.info(f"User question: {q}")
            logging.info(f"Bot answer: {response['text'][i]}")
            self.assertIsNotNone(response['text'][i])

    def test_bot_responses_to_empty_questions(self):
        questions = []
        response = post(f"{self._url}/app", data=json.dumps({"text": questions}),
                        headers={'Content-Type': 'application/json'}).json()
        self.assertEqual(questions, response['text'])

    def test_bot_responses_to_corrupted_questions(self):
        questions = ["5234dagq", "&$(%", "813eca123@42"]
        response = post(f"{self._url}/app", data=json.dumps({"text": questions}),
                        headers={'Content-Type': 'application/json'}).json()
        self.assertEqual(len(questions), len(response['text']))
        for i, q in enumerate(questions):
            logging.info(f"User question: {q}")
            logging.info(f"Bot answer: {response['text'][i]}")
            self.assertIsNotNone(response['text'][i])


if __name__ == '__main__':
    unittest.main()
