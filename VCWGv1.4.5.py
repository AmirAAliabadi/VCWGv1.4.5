from UWG import UWG

# Define the .epw, .uwg filenames to create an UWG object.
# UWG will look for the .epw file in the UWG/resources/epw folder,
# and the .uwg file in the UWG/resources/parameters folder.

epw_filename = "ERA5_Guelph_2015.epw"
param_filename = "initialize_Guelph_0.uwg"         # .uwg file name

# Initialize the UWG object and run the simulation
uwg = UWG(epw_filename, param_filename,'','','','')
uwg.run()
