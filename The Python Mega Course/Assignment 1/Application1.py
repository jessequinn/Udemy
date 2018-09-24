'''
Simple json search program.

Returns dictionary value from key given.
'''

import json
from difflib import get_close_matches


def return_dict_value(json, key):
    if key in json:
        return json[key]
    elif key.lower() in json:
        return json[key.lower()]
    elif key.upper() in json:
        return json[key.upper()]
    elif key.title() in json:
        return json[key.title()]
    elif len(get_close_matches(key, json.keys())) > 0:
        yn = input('Did you mean %s instead? Enter Y if yes, or N if no. ' % get_close_matches(key, json.keys())[0])
        if yn.upper() == 'Y':
            return json[get_close_matches(key, json.keys())[0]]
        elif yn.upper() == 'N':
            return "The word doesn't exist. Please double check it."
        else:
            return 'We didn\'t understand your query.'
    else:
        return 'The word doesn\'t exist. Please double check it.'


if __name__ == '__main__':
    with open('./data.json', 'r') as j:
        data = json.load(j)

    try:
        word = input('Enter word: ')

        output = return_dict_value(data, word)

        if len(output) > 1 and not type(output) is str:
            for o in output:
                print(o)
        else:
            print(output)

    except KeyError:
        print('Error with word.')
