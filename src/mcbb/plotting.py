import openmc 
import matplotlib.pyplot as plt
def plot_openmc_geometry(geometry : openmc.Geometry):
    '''
    Function to plot the OpenMC geometry in 1D

    Colors are automatically assigned to each material

    Parameters
    ----------
    geometry : openmc.Geometry
        The OpenMC geometry object to plot
    Returns
    -------
    None
    -------
    '''    
    colordict = {}
    colors = ['lightpink', 'chocolate', 'lightgrey', 'azure', 'black', 'olive', 'lightblue', 'lightgreen']
    for j,i in enumerate(geometry.get_all_cells().values()):
        colordict[i] = colors[j%len(colors)]
    plt.figure()
    geometry.plot(width = (223.2,100), pixels=(1116,100), colors = colordict, legend=True, axes= plt.gca(), axis_units = 'm', interpolation='none')
    plt.xlabel('x [m]')
    plt.yticks([])
    plt.tight_layout()
