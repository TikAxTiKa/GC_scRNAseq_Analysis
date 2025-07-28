import cell2cell as c2c
import liana as li

import numpy as np
import pandas as pd

data_folder = '/public/home/szu_yanwy/miniconda3/envs/zyx/TensorCell2Cell20241114/'
output_folder = '/public/home/szu_yanwy/miniconda3/envs/zyx/TensorCell2Cell20241114/tc2c-outputs/'
c2c.io.directories.create_directory(output_folder)

liana_res = pd.read_csv(data_folder + 'LIANA_by_sample.csv')

sorted_samples =list(liana_res['sample'].unique())
sorted_samples
#['GSM3954946', 'GSM3954947', 'GSM3954948', 'GSM3954949', 'GSM3954950', 
#'GSM3954951', 'GSM3954952', 'GSM3954953', 'GSM3954954', 'GSM3954955', 
#'GSM5573471', 'GSM5573472', 'GSM5573478', 'GSM5573479', 'GSM5573480', 
#'GSM5573481', 'GSM5573482', 'GSM5573483', 'GSM5573486', 'GSM5573487', 
#'GSM5573490', 'GSM5573491', 'GSM5573494', 'GSM5573497', 'GSM5573498', 
#'GSM5573499', 'GSM5573504']

tensor = li.multi.to_tensor_c2c(liana_res=liana_res, # LIANA's dataframe containing results
                                sample_key='sample', # Column name of the samples
                                source_key='source', # Column name of the sender cells
                                target_key='target', # Column name of the receiver cells
                                ligand_key='ligand_complex', # Column name of the ligands
                                receptor_key='receptor_complex', # Column name of the receptors
                                score_key='magnitude_rank', # Column name of the communication scores to use
                                non_negative = True, # set negative values to 0
                                inverse_fun=lambda x: 1 - x, # Transformation function
                                non_expressed_fill=None, # Value to replace missing values with
                                how='outer', # What to include across all samples
                                lr_fill=np.nan, # What to fill missing LRs with
                                cell_fill = np.nan, # What to fill missing cell types with
                                outer_fraction=1/3., # Fraction of samples as threshold to include cells and LR pairs.
                                lr_sep='^', # How to separate ligand and receptor names to name LR pair
                                context_order=sorted_samples, # Order to store the contexts in the tensor
                                sort_elements=True # Whether sorting alphabetically element names of each tensor dim. Does not apply for context order if context_order is passed.
                               )

tensor.shape
#(27, 275, 43, 43)

from collections import defaultdict

element_dict = defaultdict(lambda: 'Unknown')

context_dict = element_dict.copy()

context_dict.update({
	'GSM3954946':'NAG', 
	'GSM3954947':'NAG', 
	'GSM3954948':'NAG', 
	'GSM3954949':'CAG', 
	'GSM3954950':'CAG', 
	'GSM3954951':'CAG', 
	'GSM3954952':'IM', 
	'GSM3954953':'IM', 
	'GSM3954954':'IM', 
	'GSM3954955':'IM', 
	'GSM5573471':'EGC', 
	'GSM5573472':'EGC', 
	'GSM5573478':'EGC', 
	'GSM5573480':'EGC', 
	'GSM5573483':'EGC', 
	'GSM5573494':'EGC', 
	'GSM5573497':'EGC', 
	'GSM5573479':'LGC', 
	'GSM5573481':'LGC', 
	'GSM5573482':'LGC', 
	'GSM5573486':'LGC', 
	'GSM5573487':'LGC', 
	'GSM5573490':'LGC', 
	'GSM5573491':'LGC', 
	'GSM5573498':'LGC', 
	'GSM5573499':'LGC', 
	'GSM5573504':'LGC'
})
dimensions_dict = [context_dict, None, None, None]

meta_tensor = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor,
                                              metadata_dicts=[context_dict, None, None, None],
                                              fill_with_order_elements=True
                                             )
c2c.io.export_variable_with_pickle(tensor, output_folder + '/BALF-Tensor.pkl')
c2c.io.export_variable_with_pickle(meta_tensor, output_folder + '/BALF-Tensor-Metadata.pkl')

pd.__version__
#'2.1.4'











