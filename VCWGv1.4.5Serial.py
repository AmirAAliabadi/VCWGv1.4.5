from UWG import UWG
from PostProcess2Serial import PostProcess2Serial

# Define the .epw, .uwg filenames to create an UWG object.
# UWG will look for the .epw file in the UWG/resources/epw folder,
# and the .uwg file in the UWG/resources/parameters folder.

epw_filename = "ERA5_Guelph_2015.epw"
param_filename_1 = "initialize_Guelph_1.uwg"         # .uwg file name
param_filename_2 = "initialize_Guelph_2.uwg"         # .uwg file name
param_filename_3 = "initialize_Guelph_3.uwg"         # .uwg file name
param_filename_4 = "initialize_Guelph_4.uwg"         # .uwg file name
param_filename_5 = "initialize_Guelph_5.uwg"         # .uwg file name
param_filename_6 = "initialize_Guelph_6.uwg"         # .uwg file name
param_filename_7 = "initialize_Guelph_7.uwg"         # .uwg file name
param_filename_8 = "initialize_Guelph_8.uwg"         # .uwg file name
param_filename_9 = "initialize_Guelph_9.uwg"         # .uwg file name
param_filename_10 = "initialize_Guelph_10.uwg"         # .uwg file name
param_filename_11 = "initialize_Guelph_11.uwg"         # .uwg file name
param_filename_12 = "initialize_Guelph_12.uwg"         # .uwg file name

# Initialize the UWG object and run the simulation

uwg = UWG(epw_filename, param_filename_1,'','','','')
uwg.run()
PostProcess2Serial(1, "Output/Perf-Metrics-Case-8-Jan.txt")

uwg = UWG(epw_filename, param_filename_2,'','','','')
uwg.run()
PostProcess2Serial(1, "Output/Perf-Metrics-Case-8-Feb.txt")

uwg = UWG(epw_filename, param_filename_3,'','','','')
uwg.run()
PostProcess2Serial(1, "Output/Perf-Metrics-Case-8-Mar.txt")

uwg = UWG(epw_filename, param_filename_4,'','','','')
uwg.run()
PostProcess2Serial(1, "Output/Perf-Metrics-Case-8-Apr.txt")

uwg = UWG(epw_filename, param_filename_5,'','','','')
uwg.run()
PostProcess2Serial(0, "Output/Perf-Metrics-Case-8-May.txt")

uwg = UWG(epw_filename, param_filename_6,'','','','')
uwg.run()
PostProcess2Serial(0, "Output/Perf-Metrics-Case-8-Jun.txt")

uwg = UWG(epw_filename, param_filename_7,'','','','')
uwg.run()
PostProcess2Serial(0, "Output/Perf-Metrics-Case-8-Jul.txt")

uwg = UWG(epw_filename, param_filename_8,'','','','')
uwg.run()
PostProcess2Serial(0, "Output/Perf-Metrics-Case-8-Aug.txt")

uwg = UWG(epw_filename, param_filename_9,'','','','')
uwg.run()
PostProcess2Serial(0, "Output/Perf-Metrics-Case-8-Sep.txt")

uwg = UWG(epw_filename, param_filename_10,'','','','')
uwg.run()
PostProcess2Serial(1, "Output/Perf-Metrics-Case-8-Oct.txt")

uwg = UWG(epw_filename, param_filename_11,'','','','')
uwg.run()
PostProcess2Serial(1, "Output/Perf-Metrics-Case-8-Nov.txt")

uwg = UWG(epw_filename, param_filename_12,'','','','')
uwg.run()
PostProcess2Serial(1, "Output/Perf-Metrics-Case-8-Dec.txt")

