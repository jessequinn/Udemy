import os

root = 'lesson_193_music'

for path, directories, files in os.walk(root, topdown=True):
    if files:
        print(path)
        first_split = os.path.split(path)
        print(first_split)
        print("*"*40)
        second_split = os.path.split(first_split[0])
        print(second_split)
        files.sort()
        for f in files:
            song_details = f[:-5].split(' - ')
            print(song_details)
        print("*"*40)


    # print(directories)
    # print(files)
    # input()

    # for f in files:
    #     print('\t{}'.format(f))
