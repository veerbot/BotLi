import fileinput

file_path = "./engines/supraengine"

for line in fileinput.input(file_path, inplace=True):
    print(line.replace("zobrist_hash()", "transposition_key"), end='')
