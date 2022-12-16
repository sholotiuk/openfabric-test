from utils import *
import argparse
import json

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    with open("config/execution.json", "w") as jsonfile:
        config_object = json.load(jsonfile)
        config_object["config_class"] = configuration
        json.dump(config_object, jsonfile)
        jsonfile.close()


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    # Configuration file parsing
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        type=str,
        default='model_params/cpu_params.cfg',
        help="Path to the config"
    )
    args = arg_parser.parse_args()
    config_path = args.config
    kwargs = parse_config(config_path)

    # Extract parameters
    general_params = kwargs.get('general_params', {})
    device = general_params.get('device', -1)
    seed = general_params.get('seed', None)
    debug = general_params.get('debug', False)

    generation_pipeline_kwargs = kwargs.get('generation_pipeline_kwargs', {})
    generation_pipeline_kwargs = {**{
        'model': 'microsoft/DialoGPT-medium'
    }, **generation_pipeline_kwargs}

    generator_kwargs = kwargs.get('generator_kwargs', {})
    generator_kwargs = {**{
        'max_length': 1000,
        'do_sample': True,
        'clean_up_tokenization_spaces': True
    }, **generator_kwargs}

    prior_ranker_weights = kwargs.get('prior_ranker_weights', {})
    cond_ranker_weights = kwargs.get('cond_ranker_weights', {})

    chatbot_params = kwargs.get('chatbot_params', {})
    max_turns_history = chatbot_params.get('max_turns_history', 2)

    # Prepare the pipelines
    generation_pipeline = load_pipeline('text-generation', device=device, **generation_pipeline_kwargs)
    ranker_dict = build_ranker_dict(device=device, **prior_ranker_weights, **cond_ranker_weights)
    logger.info("Running the bot...")
    start_message()
    turns = []
    output = []
    for text in request.text:
        print("User: ", text)
        if max_turns_history == 0:
            turns = []
        if text.lower() == '/start':
            start_message()
            turns = []
            continue
        if text.lower() == '/reset':
            reset_message()
            turns = []
            continue
        if text.startswith('/'):
            print('Command not recognized.')
        # A single turn is a group of user messages and bot responses right after
        turn = {
            'user_messages': [],
            'bot_messages': []
        }
        turns.append(turn)
        turn['user_messages'].append(text)
        # Merge turns into a single prompt (don't forget delimiter)
        prompt = ""
        from_index = max(len(turns) - max_turns_history - 1, 0) if max_turns_history >= 0 else 0
        for turn in turns[from_index:]:
            # Each turn begins with user messages
            for user_message in turn['user_messages']:
                prompt += clean_text(user_message) + generation_pipeline.tokenizer.eos_token
            for bot_message in turn['bot_messages']:
                prompt += clean_text(bot_message) + generation_pipeline.tokenizer.eos_token

        # Generate bot messages
        bot_messages = generate_responses(
            prompt,
            generation_pipeline,
            seed=seed,
            debug=debug,
            **generator_kwargs
        )
        if len(bot_messages) == 1:
            bot_message = bot_messages[0]
        else:
            bot_message = pick_best_response(
                prompt,
                bot_messages,
                ranker_dict,
                debug=debug
            )
        print("Bot:", bot_message)
        turn['bot_messages'].append(bot_message)

        response = bot_message
        output.append(response)

    return SimpleText(dict(text=output))


def start_message():
    print("Bot:",
          "Let's have a conversation about science! "
          "Type something... "
          "\nIf I'm getting annoying, type \"/reset\". ")


def reset_message():
    print("Bot:", "Oh, I'm reset! See you later. ")
