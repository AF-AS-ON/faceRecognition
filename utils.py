filename = "each pair scoring.csv"
directory = "faces_for_clustering"

import os


def parse_csv_and_folders():
    import csv

    with open(filename,'r') as file:
        csv_reader = csv.reader(file)
        face_pairs = set()
        first = True
        for row in csv_reader:
            if first:
                first = False
                continue
            face_pairs.add(tuple(sorted((row[0],row[1]))))
        print(sorted(face_pairs))

    pic_pairs = set()
    for dir_name in os.listdir(directory):
        pic1, pic2 = os.listdir(f"{directory}/{dir_name}")
        pic1 = pic1.split(".")[0]
        pic2 = pic2.split(".")[0]
        pic_pairs.add(tuple(sorted((pic1,pic2))))


    print(sorted(pic_pairs))
    pic_pairs.remove(("AAO","ADD"))
    assert pic_pairs == face_pairs

    return pic_pairs

def execute(command, exit_on_non_zero=True):
    import subprocess
    return subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print("Finished running")


def our_main(arch):
    for dir_name in os.listdir(directory):
        # import subprocess
        import image_similarity_2
        image_similarity_2.main(arch, f"{directory}/{dir_name}", "jpg", False)
        # command_template  = f"python {directory}/image_similarity_2.py {directory}/{dir_name}"
        # result = subprocess.check_output(command_template).decode()
        # result = os.system(command_template)
        # print(result)

our_main('alexnet')