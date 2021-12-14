import matplotlib.pyplot as plt

def plot_metric(model_training_history):
    """
    Plots all metrics from a training history object
    
    Arguments 
    ---------
        model_training_history: object
            A history object containing accuracy, loss, validation accuracy, and validation loss
            for each epoch trained 
    """   
    
    # Get list of metrics from histroy object 
    keys = list(model_training_history.history.keys())
    
    # Iterate over first half of metrics (accuracy and loss) 
    for metric in keys[:2]: 
        # Get metric 2 (validaiton version of current metric) 
        metric_2 = 'val_' + metric
        
        # Get values for each metric 
        metric_value_1 = model_training_history.history[metric]
        metric_value_2 = model_training_history.history[metric_2]
        
        # Calculate number of epochs trained 
        epochs = range(len(metric_value_1)) 
        
        # Create plot name from 2 metrics 
        name = metric + '_vs_' + metric_2
        
        # Plot the graph of the 2 metrics
        print_plot(name, metric, metric_2, metric_value_1, metric_value_2, epochs)
    
def print_plot(plot_name, metric_1, metric_2, metric_value_1, metric_value_2, epochs): 
    """
    Helper function for plot_metric that performs graph plotting 
    
    Arguments 
    ---------
        plot_name: string
            The name of the graph (and subsequent save file) 
        metric_1: string 
            The name of the first metric 
        metric_2: 
            The name of the second metric 
        metric_value_1: float 
            The value of the first metric 
        metric_value_2: float   
            The value of the second metric 
        epochs: int
            Number of epochs trained 
    """
    # Clear the Graph
    plt.clf()
    
    # Plot the Graph
    plt.plot(epochs, metric_value_1, 'blue', label = metric_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_2)

    # Add title to the plot
    plt.title(str(plot_name))

    # Add legend to the plot
    plt.legend()

    # Save plot to file
    plt.savefig("{}.png".format(plot_name))

def print_time(s, secs):
    """
    Prints to the console a time value in the format #m #.##s 
    
    Arguments
    ---------
        s: String 
            Identifier string to output with time (i.e. Start)
        secs: float
            The time value in seconds to be output
    """
    minutes = int(secs / 60)
    seconds = secs - minutes * 60
    print("{}:".format(s))
    print("{m}m {s:.2f}s".format(m=minutes,s=seconds))