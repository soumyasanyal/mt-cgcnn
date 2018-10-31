import csv, os, json, math, sys
from pymatgen import MPRester
import numpy as np

# key to query data from Materials Project as string
API_KEY = "<WRITE MATERIALS PROJECT API KEY HERE>"
# path where files are written
FILEPATH = "data/sample/"


class Logger(object):
	"""Writes both to file and terminal"""
	def __init__(self, savepath, mode="a"):
		self.terminal = sys.stdout
		self.log = open(savepath + "logfile.log", mode)

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass


def checkNull(dummy_dictionary):
	"""Given a dictionary, checks if any of the values is null"""
	if None in list(dummy_dictionary.values()):
		return True
	else:
		return False


def queryMPDatabse(input_filepath, properties, sample_fraction=1, fetch_cif=True, fetch_mpid=True, print_data=False):
	"""
	Query the Materials Project Database

	Parameters
	----------

	input_filepath: str
		Filepath containing list of material ids to be queried.
	properties: list
		The properties to query from the database e.g. ["formation_energy_per_atom", "band_gap"]
		NOTE: these are the properties we're trying to model the code to predict. For cif file or 
		material_id, use the other parameters below.
	sample_fraction: float
		After reading the input_filepath, sample_fraction of datapoints are selected and data for these
		samples are fetched from the database.
	fetch_cif: bool
		If true, fetches the cif data for the material
	fetch_mpid: bool
		if true, fetches the material_id for the material
	print_data: bool
		Print data in nice json format

	Returns
	-------

	dataset:
		Data corresponding to the query
	"""
	materials_id_list = []
	with open(input_filepath, 'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		for row in csv_reader:
			materials_id_list.append(str(row[0]))
	materials_id_list = np.random.choice(np.array(materials_id_list),\
							math.ceil(len(materials_id_list)*sample_fraction), replace=False).tolist()
	with MPRester(API_KEY) as m:
		print("Querying Materials Project Database...")
		query = {"material_id": {"$in": materials_id_list}}
		if fetch_cif:
			properties.append("cif")
		if fetch_mpid:
			properties.append("material_id")
		dataset = m.query(query, properties)
		if print_data:
			print(json.dumps(dataset, sort_keys=True, indent=4, separators=(',', ': ')))
		print("Done Querying. Fetched data for ", str(len(dataset)), " crystals")

	return dataset

def processData(dataset, has_cif=True, has_mpid=True, total_size=None):
	"""
	Process the data and write the required data in files. Implicitly filter out
	data where any one of the property value is missing
	
	Parameters
	----------
	dataset:
		Dataset containing the fetched data stored rowwise
	has_cif: bool
		If true, dataset contains the cif data for the material
	has_mpid: bool
		if true, dataset contains the material_id for the material
	total_size: int
		The maximum size of the dataset to be created
	"""

	print("Processing dataset...")
	material_hash_counter = 0	# unique counter to identify each material
	material_id_hash_list = []
	idprop_list = []
	cif_list = []

	# hack for the case when total_size is None
	if total_size is None:
		total_size = len(dataset)

	for row in dataset:
		property_keys = [*row]		# list of keys
		property_values = list(row.values())	# list of values
		missing_data = True if None in property_values else False
		if not missing_data and total_size > 0:
			if has_cif:
				cif = row["cif"]
				cif_list.append([material_hash_counter, cif])
				property_keys.remove("cif")
			if has_mpid:
				material_id = row["material_id"]
				material_id_hash_list.append([material_hash_counter, material_id])
				property_keys.remove("material_id")
			fetched_properties = [row[x] for x in property_keys]
			idprop_list.append([material_hash_counter] + fetched_properties)
			total_size -= 1
			material_hash_counter += 1
	print("Finished processing dataset!")
	print("Writing data to files...")
	writeData(idprop_list, cif_list, material_id_hash_list)


def writeData(idprop_list, cif_list, material_id_hash_list):
	"""
	Write data to files in a specific required structure format

	Parameters
	----------
	idprop_list: list
		List containing unique material id and the properties fetched (no None cases)
	cif_list: list
		List containing unique material id and the cif file
	material_id_hash_list: list
		List containing unique material id and the material_id from Materials Project DB
	"""
	if not os.path.exists(FILEPATH):
		os.makedirs(FILEPATH)

	# Write id_prop.csv
	with open(FILEPATH + '/id_prop.csv', 'w') as file:
		writer = csv.writer(file)
		writer.writerows(idprop_list)
	print("Written id_prop.csv")

	# Write the cif files
	for row in cif_list:
		unique_id, cif = row
		with open(FILEPATH + '/' + str(unique_id) + '.cif', 'w') as file:
			file.write(cif)
		print("Written " + str(unique_id) + ".cif")

	# Write materials id hash map:
	# This contains the Unique ID and the material_id (as obtained from original dataset)
	if len(material_id_hash_list):
		with open(FILEPATH + '/material_id_hash.csv', 'w') as file:
			writer = csv.writer(file)
			writer.writerows(material_id_hash_list)
	print("Written material_id_hash.csv")


if __name__ == '__main__':
	dataset = queryMPDatabse('mpids_sample.csv', ["formation_energy_per_atom", "band_gap"],\
					print_data=False, fetch_cif=True, sample_fraction=1)
	processData(dataset, has_cif=True)
