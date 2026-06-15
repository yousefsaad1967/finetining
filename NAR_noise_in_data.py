'''
Apache 2.0 License. This source code builds on source code from the NILMTK project.

Christoph Klemenjak 2019. Please cite our paper (and the NILMTK paper) in case you use this code: http://makonin.com/doc/ISGT-NA_2020b.pdf
https://github.com/klemenjak/comparability
'''

import warnings
import numpy as np
from nilmtk import DataSet
from nilmtk.elecmeter import ElecMeterID

# 1. Suppress pandas and PyTables warnings to keep the terminal output clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def noise_aggregate_ratio(elec_meter, power_type='active', meterkeys=None, good_sections_only=True):
    '''
    Computes the noise-to-aggregate ratio (NAR) of an elec metergroup. 
    For more information, see our paper: http://makonin.com/doc/ISGT-NA_2020b.pdf
    :param elec_meter: elec of a dataset
    :param meterkeys: either None or array that contains meter IDs
    :param power_type: AC power type, either active or apparent
    :param good_sections_only: option to use good sections only
    :return: NAR ratio (float)
    '''
    mains = elec_meter.mains()

    if not meterkeys:
        submeters = elec_meter.meters_directly_downstream_of_mains()
    else:
        submeters = elec_meter.from_list(
            [ElecMeterID(elec_meter[m].instance(), elec_meter.building(), elec_meter.dataset()) for m in meterkeys])

    if good_sections_only:
        good_mains_sections = mains.good_sections()
        loader_kwargs = {'sections': good_mains_sections}
    else:
        loader_kwargs = {}

    mains_total_energy = mains.total_energy(**loader_kwargs)
    mains_ac_types = mains_total_energy.keys()

    proportion = np.float64(0.0)

    for meter in submeters.meters:
        try:
            meter_total_energy = meter.total_energy(**loader_kwargs)
        except KeyError:
            warnings.warn('KeyError at '+str(meter))
            return [-1, 'power type: '+ power_type, 'good sections only: ' +str(good_sections_only)]
        
        meter_ac_types = meter_total_energy.keys()
        shared_ac_types = set(mains_ac_types).intersection(meter_ac_types)

        if len(shared_ac_types) > 1:
            proportion += meter_total_energy[power_type] / mains_total_energy[power_type]
            continue

        elif len(shared_ac_types) == 1 and shared_ac_types.__contains__(power_type):
            ac_type = list(shared_ac_types)[0]
            proportion += meter_total_energy[ac_type] / mains_total_energy[ac_type]
            continue

        elif len(shared_ac_types) == 1 and not shared_ac_types.__contains__(power_type):
            warnings.warn('No matching power types found!')
            continue

        elif len(shared_ac_types) == 0:
            warnings.warn('No matching power types found!')
            continue

    return round(1 - proportion, 2)


# ==========================================
# Main Execution Area
# ==========================================
if __name__ == "__main__":
    dataset_path = 'REFIT_NILMTK.h5'
    
    print("Loading dataset...")
    dset = DataSet(dataset_path)

    # Select data for House 2
    elec = dset.buildings[5].elec

    print("Calculating NAR (Active Power) for House 5...")
    nar_active = noise_aggregate_ratio(elec, power_type='active')
    
    print("-" * 40)
    print(f"NAR (Active Power) for House 5: {nar_active * 100} %")
    print("-" * 40)

    # 2. Close the dataset safely to avoid UnclosedFileWarning
    dset.store.close()
    print("Dataset closed successfully.")