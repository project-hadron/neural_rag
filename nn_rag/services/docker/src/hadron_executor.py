from nn_rag import Controller
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

__author__ = 'Darryl Oatridge'


def run_hadron_controller():
    # get the controller run attributes from Environment
    uri_pm_repo = os.environ['HADRON_DOMAIN_REPO_PATH']
    if not uri_pm_repo:
        raise KeyError("The mandatory environment variable 'HADRON_DOMAIN_REPO_PATH' has not been set")

    run_book = None if not os.getenv('HADRON_CONTROLLER_RUNBOOK') else os.getenv('HADRON_CONTROLLER_RUNBOOK')
    repeat = None if not os.getenv('HADRON_CONTROLLER_REPEAT') else os.getenv('HADRON_CONTROLLER_REPEAT')
    sleep = None if not os.getenv('HADRON_CONTROLLER_SLEEP') else os.getenv('HADRON_CONTROLLER_SLEEP')
    run_time = None if not os.getenv('HADRON_CONTROLLER_RUNTIME') else os.getenv('HADRON_CONTROLLER_RUNTIME')
    run_cycle_report = None if not os.getenv('HADRON_CONTROLLER_REPORT') else os.getenv('HADRON_CONTROLLER_REPORT')
    source_check_uri = None if not os.getenv('HADRON_CONTROLLER_SOURCE_CHECK') else os.getenv(
        'HADRON_CONTROLLER_SOURCE_CHECK')

    # instantiate the Controller passing any remaining kwargs
    controller = Controller.from_env(uri_pm_repo=uri_pm_repo, default_save=False, has_contract=True)
    # run the arrows nano services.
    controller.run_controller(run_book=run_book, repeat=repeat, sleep=sleep, run_time=run_time,
                              source_check_uri=source_check_uri, run_cycle_report=run_cycle_report)


def set_env_from_payload_if_available():
    payload = {
        "hadron_kwargs":{
            "HADRON_DOMAIN_REPO_PATH": "https://raw.githubusercontent.com/project-hadron/neural_rag/main/jupyter/knowledge/hadron/contracts/",
            "HADRON_KNOWLEDGE_SOURCE_URI": "https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.pdf",
            "HADRON_KNOWLEDGE_PERSIST_URI": "./cache/",
            "HADRON_PROFILE_NAME": "EU_AI_Act_2024",
        }
    }

    # extract any extra kwargs
    hadron_kwargs = payload.get('hadron_kwargs', {})
    # export and pop any environment variable from the kwargs
    for key in tuple(hadron_kwargs.keys()):
        key = key.upper()
        os.environ[key] = hadron_kwargs.pop(key)

if __name__ == '__main__':
    set_env_from_payload_if_available()
    run_hadron_controller()
