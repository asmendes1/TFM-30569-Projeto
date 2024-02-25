
import csv
import os



class CsvFile:
    def __init__(self, path, mode):
        self.path = path #self.path = "../" + path
        self.mode = mode

        if self.mode == "r":
            header = self.get_header_from_file()
            self.__header = header
        else:
            self.__header = None
            # self.write_lines_on_file([[""]])

    def get_header_from_file(self):
        with open(self.path, "r") as file:
            csvreader = csv.reader(file)
            # Extract the field names
            header = next(csvreader)
        return header

    def get_header(self):
        return self.__header

    def get_all_data(self):
        with open(self.path, "r") as file:
            # Create a reader
            csvreader = csv.reader(file)
            # Extract the rows/records
            rows = []
            for row in csvreader:
                rows.append(row)
        return rows[3:]  # ignoring the header

    # arr: indices das colunas qua se pretende
    # obter [inicio, fim(inclusive)]
    def get_columns(self, arr):
        ini_idx = arr[0]
        final_idx = arr[1]
        with open(self.path, "r") as file:
            # Create a reader
            csvreader = csv.reader(file)
            # Extract the rows/records
            rows = []
            for row in csvreader:
                splited_row = row[ini_idx:final_idx][0].split(";")
                selected_columns = splited_row[ini_idx:final_idx + 1]
                rows.append(selected_columns)
                #print("ROW: ", row)
                #print("ROW SELECTED COLUMNS: ", selected_columns)
        return rows[1:] # ignoring the header

    # data é uma lista de listas ou um array de arrays
    def write_one_line_on_file(self, line):
        with open (self.path, "a", newline="") as file:
            # create a writer
            writer = csv.writer(file)
            # write a line
            writer.writerow(line)

    # data é uma lista de listas ou um array de arrays
    def write_lines_on_file(self, data):
        with open (self.path, "a", newline="") as file:
            # cria um writer
            writer = csv.writer(file)
            # escreve várias linhas
            writer.writerows(data)

    def clear_file(self):
        with open(self.path, 'r+') as file:
            file.truncate(0)

    @staticmethod
    def remove_file(path):
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
            print("file '" + path + "' has been deleted")
        else:
            print("file '" + path + "' not found")

