import pandas as pd
import os
import json

def merge_logfile_data(path:str, nareas:int=32):
    """Collects the data contained in the logfiles and merge them
    into a single csv file.

    Args:
        path (str): path to the directory containing the simulations results.
        nareas (int, optional): Number of areas simulated. Defaults to 32.
    """

    sim_dirs = os.listdir(path)

    sim_dirs = [dir for dir in sim_dirs if os.path.isdir(path+dir+"/recordings/")==True]
    
    for dir in sim_dirs:
        if(os.path.isdir(path+dir)):
            if(os.path.isfile(path+dir+"/recordings/"+dir+"_merged.csv")):
                #print("Data already merged!")
                pass
            else:
                print("Merging data for simulation {}".format(dir))
                sim_res = pd.DataFrame({})
                for area in range(nareas):
                    fn = dir+"_logfile_"+str(area)
                    with open(path+dir+"/recordings/"+fn) as f:
                        area_logfile = json.loads(f.read())
                    area_logfile = pd.DataFrame(area_logfile, index=[area])
                    area_logfile['total_constr'] = area_logfile['time_configure'] + area_logfile['time_network_local_tot'] + area_logfile['time_connect_global'] + area_logfile['time_calibrate']
                    sim_res = pd.concat([sim_res, area_logfile])

                sim_res.to_csv(path+dir+"/recordings/"+dir+"_merged.csv")
            
                print("Merging done")
        else:
            pass


def process_sim_data(path:str, nsim:int=20, method:str='none', **kwargs):
    """Gets the values from each simulation and average them over the MPI processes.
    Finally, the values for each simulation are collected into a csv file.

    Args:
        path (str): path to the directory containing the simulations results.
        nsim (int, optional): Number of simulations to be selected. Defaults to 20.
        method (str, optional): method for averaging the benahmarking results.
            over the MPI processes. If 'mean', the average of the time values collected.
            for each process is used, if 'max' the maximum value for each timer is taken. 
            If none, it does not average the data. Defaults to 'none'.
        **kwargs: Arbitrary keyword arguments.
               - nareas (int): Number of areas simulated.
    Returns:
        _type_: _description_
    """

    METHODS = ['max', 'mean', 'none']

    nareas = int(kwargs.get('nareas', 32))
    merge_logfile_data(path=path, nareas=nareas)
    
    sim_dirs = os.listdir(path)

    sim_dirs = [dir for dir in sim_dirs if os.path.isdir(path+dir+"/recordings/")==True]

    if(nsim>len(sim_dirs)):
        raise ValueError("Number of simulations selected exceeds the available simulations. Please choose nsim<={}".format(len(sim_dirs)))
    else:
        sim_dirs = sim_dirs[:nsim]

    all_sim_data = pd.DataFrame({})
    for ndir, dir in enumerate(sim_dirs):
        sim_data = pd.read_csv(path+dir+"/recordings/"+dir+"_merged.csv", index_col=0)
        if(method=='max'):
            # should we choose the MPI proc with the overall max time or
            # the max of each timer indipendently?
            sim_data = sim_data.max(axis=0).to_dict()
            sim_data["sim_label"] = dir
            all_sim_data = pd.concat([all_sim_data, pd.DataFrame(sim_data, index=[ndir])])
        elif(method=='mean'):
            sim_data = sim_data.mean(axis=0).to_dict()
            sim_data["sim_label"] = dir
            all_sim_data = pd.concat([all_sim_data, pd.DataFrame(sim_data, index=[ndir])])
        elif(method=='none'):
            sim_data["sim_label"] = dir
            all_sim_data = pd.concat([all_sim_data, sim_data])
        else:
            raise ValueError("Method not found, please chose one from the following: {}".format(METHODS))

    all_sim_data.to_csv(path+"all_sim_data_"+method+".csv")

    return all_sim_data


def prepare_data_for_plot(paths:list, labels:list, nareas:list, **kwargs):

    nsim = int(kwargs.get('nsim', 10))
    method = str(kwargs.get('method', 'mean'))
    save_file = bool(kwargs.get('save_file', True))

    data = pd.DataFrame({})

    for nodes, p in enumerate(paths):
        dum = process_sim_data(path=p, nsim=nsim, method=method, nareas=nareas[nodes])
        dum = dum.drop(["sim_label"], axis=1)

        dum = dum / 1e9

        dum["nodes"] = [labels[nodes] for i in range(len(dum))]

        if labels[nodes]<16:
            gpus = 4
        elif labels[nodes]==16:
            gpus = 2
        elif labels[nodes]==32:
            gpus = 1

        print(dum)
        dum["gpus_per_node"] = [gpus for i in range(len(dum))]

        dum["model_time_sim"] = [10.0 for i in range(len(dum))]

        dum['time_create_nodes'] = (dum['time_create_neurons'].values +
                                        dum['time_create_devices'].values)
        
        dum['time_connect_neurons'] = dum['time_connect_local']
        
        dum['time_connect_local'] = (dum['time_connect_neurons'].values +
                                        dum['time_connect_devices'].values)
        
        dum['time_area_packing'] = (dum['time_network_local_tot'].values - 
                                        dum['time_create_nodes'].values - 
                                        dum['time_connect_local'].values)
        
        dum['sim_factor'] = (dum['time_simulate'].values / dum['model_time_sim'].values)

        dum = dum[['nodes', 'gpus_per_node', 'model_time_sim', 
                   'time_configure', 'time_create_neurons', 'time_create_devices', 'time_connect_neurons', 'time_connect_devices',
                   'time_create_nodes', 'time_area_packing', 'time_connect_local', 'time_network_local_tot', 'time_connect_global', 'time_calibrate',
                   'total_constr', 'time_presimulate', 'time_simulate', 'sim_factor']]

        data = pd.concat([data, dum])
    
    data = data.reset_index(drop=True)

    print(data)

    dict_ = {'nodes': 'first',
            'gpus_per_node': 'first',
            'model_time_sim': 'first',
            'time_configure': ['mean', 'std'],
            'time_create_nodes': ['mean', 'std'],
            'time_connect_local': ['mean', 'std'],
            'time_network_local_tot': ['mean', 'std'],
            'time_area_packing': ['mean', 'std'],
            'time_connect_global': ['mean', 'std'],
            'total_constr': ['mean', 'std'],
            'time_calibrate': ['mean', 'std'],
            'time_presimulate': ['mean', 'std'],
            'time_simulate': ['mean', 'std'],
            'sim_factor': ['mean', 'std'],}

    col = ['nodes', 'gpus_per_node',
            'model_time_sim',
            'time_configure', 'time_configure_std',
            'time_create_nodes', 'time_create_neurons_std',
            'time_connect_local', 'time_connect_local_std',
            'time_network_local_tot', 'time_network_local_tot_std',
            'time_area_packing', 'time_area_packing_std',
            'time_connect_global', 'time_connect_global_std',
            'total_constr', 'total_constr_std',
            'time_calibrate', 'time_calibrate_std',
            'time_presimulate', 'time_presimulate_std',
            'time_simulate', 'time_simulate_std',
            'sim_factor', 'sim_factor_std']
    
    print(data.groupby(
            ['nodes',
             'gpus_per_node',
             'model_time_sim'], as_index=False).agg(dict_))

    data.groupby(
            ['nodes',
             'gpus_per_node',
             'model_time_sim'], as_index=False).agg(dict_).to_csv("averaged_data.csv")

    if save_file:
        data.to_csv("processed_times_"+ method +".csv")

    return(data)


path2 = "./nodes-2/"
path4 = "./nodes-4/"
path6 = "./nodes-6/"
path8 = "./nodes-8/"
path16 = "./nodes-16/"
path32 = "./nodes-32/"

paths = [path2, path4, path6, path8, path16, path32]

labels = [2, 4, 6, 8, 16, 32]

nareas = [8, 16, 24, 32, 32, 32]

prepare_data_for_plot(paths=paths, labels=labels, nareas=nareas, nsim=10, method='mean')
