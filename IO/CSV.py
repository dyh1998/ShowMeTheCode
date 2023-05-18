import csv


def read_csv(filename):
    data = []
    with open(filename, 'r', encoding='UTF8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data


def write_file_handler_to_csv(file_handler, file_save_path):
    with open(file_save_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(file_save_path)
        for d in file_handler:
            writer.writerow(d)
    print("write finished")
