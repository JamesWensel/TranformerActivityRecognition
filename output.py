#! /usr/bin/python3

import os
import csv
import math
import matplotlib.pyplot as plt

### Add ability to store a single time / mem usage for an identifier and have it be useable 
### * use most recently stored data
### Add ability to print sub process times / memory usage
### Track memory / time from within class (make the call to the object pass only an identifier) 

def printProgressBar (iteration, total, prefix = '', suffix = 'Complete', decimals = 1, fill = "\u2588", unfill=' ',eol = "\r"):
    """
    Prints a progress bar when called in a loop
    
    Arguments
    ---------
    iteration: int
        Current itteration number (Used to find percent completion)
    total: int
        total iterations (Used to find percent completion) 
    prefix: str (default '') 
        prefix string to print before progress bar
    suffix: str (default '') 
        suffix string to print after progress bar
    decimals: int (default 1)
        number of decimals of percent to display
    length: int (default 100)
        character length of bar
    fill: str (default ⬛)       
        bar fill character
    unfill: str (default ' ')
        character to use for unfilled section of progress bar
    eol: str (default '\r')
        end character (e.g. "\r", "\r\n") (Str)
    """
    # Define max prefix size (determined by UCF 101 dataset class names) 
    MAX_LEN = 18
    
    # Creates percent string to print after progress bar
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    
    # Give progress bar a dynamic size, with max size being 100px Decimal + 4 for percent in shape (100.0%), 10 for other formatting
    length = os.get_terminal_size().columns - (MAX_LEN + len(suffix) + (decimals + 4) + 10)
    
    # Set max length size to 150
    length = min(150, length)
    
    # Determines whole number length of progres bar (rounds down)
    filled_length = int(length * iteration // total)
    
    # Determines remaining length of progress bar (% 1 for decimal remainder, int( ) and * 8 to force between 0-7 for unicode list
    part_length = int(((length * iteration // total) % 1) * 8)
    
    # Unicode character correspoding to partial section of progress bar
    part_char = [" ", "u\2589", "u\258A", "u\258B", "u\258C", "u\258D", "u\258E", "u\258F"][part_length]
    
    # Create progress bar string
    bar = fill * filled_length + unfill * (length - filled_length)
    
    # Create string for prefix including spaces to make all progress bars start at the same location
    pre = " " * 2 + prefix + " " * (MAX_LEN - len(prefix)) + ":" if prefix != '' else ''
    
    # Create string for suffix including spaces to make all progress bars the same length
    suf = "%" + " " * (((decimals+4) - len(percent)) + 1) + suffix + " " * 2
    
    # Print progress bar string with prefix, percent, and suffix if provided. \n eol if progress bar is done
    print(f'\r{pre}[{bar}]{percent}{suf}', end = (eol if iteration != total else '\n'))


def writeTestResults (history, filename):
    """
    Writes training history to csv
    
    Arguments
    ---------
        history: dictionary 
            contains all training results
        filename: string
            directory and file name to save results to as a csv
    
    """
    # Convert filename to a .csv filename if it is not already and add specifier
    root, ext = os.path.splitext(filename)
    if not ext or ext != '.csv': 
        ext = '.csv'
    filename = root + '_results' + ext
    
    # Create csv_data from stored history (must be in list of dicts in form [{key0: val0, key1: val1}, {key0: val2, key1: val3}])
    csv_data = []
    
    # Create empty dictionarys first 
    for i in range(len(list(history.values())[0])):
        csv_data.append(dict()) 
    
    # Fill in data for each dictionary
    for key in history: 
        for i, val in enumerate(history[key]): 
            csv_data[i][key] = val
    
    # Try to write to csv, and catch IOError
    try: 
        # Open file to write, no newline delimiter to keep from added unnecessary blank lines, and write all data
        with open(filename, 'w', newline="") as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=list(history.keys()))
            writer.writeheader()
            for row in csv_data: 
                writer.writerow(row)
    except IOError: 
        print("Error, Could Not Write CSV")


def printTestResultsGraph (key, data, filename, directory):
    """
    Prints graphs for training results
    
    Arguments
    ---------
        key: string
            specific metric from saved from the current test
        data: list of ints 
            values of the metric after each epoch of the test 
        filename: string 
            combination of test value and attribute being tested  
        directory: string
            name of the directory to save the plot in
    
    """
    # Clear the Graph
    plt.clf()

    # Plot the Graph
    plt.plot(range(len(data)), data, 'red', label = key)
    
    # Add title to the plot
    plt.title(filename + ' ' + key)

    # Add legend to the plot
    plt.legend()

    # Save plot to file
    plt.savefig(os.path.join(directory, f'{filename}_{key}.png'))
    

def outputTestResults (history, model, test, value): 
    """
    Outputs results of a single test
    
    Arguments
    ---------
        history: model history object 
            results of tests performed
        model: string 
            type of model used for this test
        test: string
            specific attribute being tested 
        value: int
            value of attribute in model for current test
    """
    
    # Create directory name and filename from current model, test, and value
    dir = f'Model_Tests/{model}/{test}'
    filename = f'{value}_{test}'
    
    # Get dictionary of stored model training history
    hist = history.history
    
    # Write all test results to csv file 
    writeTestResults(hist, os.path.join(dir, filename))
    
    # Print each specific metric tracked to its own graph
    for key in hist: 
        printTestResultsGraph(key, hist[key], filename, dir) 
        

class Output: 
    """
    Class to handle all storage of timing and memory data, and all data to be written to csv
    """
    
    def __init__ (self, header=None, initial_data=None, log_level=1): 
        """
        Initilizes Output Class
        
        Arguments
        ---------
            header: list of strings (default None)
                list containing string string to be used as CSV header (if None, sets header to ['Step', 'Time', 'Memory']
                If provided, csv_data will not be handled internally
            initial_data: list of dictionaries (default None)
                list containing initial csv data 
            log_level: int (default 1)
                determines amount of data output. 0: none, 1: Mem/Time usage of each process, 2: More Mem Data, 3: Time Elapsed
                Also determines how much data is output after program completion
        """
        # Whether or not the class will handle creating the csv data
        self.create_csv_data = True
        
        # 0 = no text output, csv output
        # 1 = Memory text before and after, Time usage of each process
        # 2 = Memory usage by each process
        # 3 = Time elapsed
        self.log_level = min(3, max(0, log_level))
        
        # Initial dictionary holding time and memory data
        self.data = dict()
        
        # Empty CSV data and CSV header for CSV data storage
        self.csv_data, self.head = [], [] 
        
        # Update header with correct values 
        if header is not None: 
            self.header = header
            self.create_csv_data = False
        else: 
            self.header = ['Step', 'Time', 'Memory']
            
        # Add any initial data provided
        if initial_data is not None:
            self.csv_data.append(initial_data)
    
    def append_csv(self, data): 
        """
        Adds data dictionary to csv data list if it matches the criteria
        
        Arguments
        ---------
            data: dictionary 
                Dictionary with data to write to CSV (must have keys == self.header)
        """
        
        # Verifies the data is in the correct CSV form (keys are equal to self.header values)
        # Appends the data to the csv, with andy Time and Memory data stored as a string
        if list(data.keys()) == self.header:
            if 'Time' in data and type(data['Time']) != str: 
                data['Time'] = self.time_string(data['Time'])
            if 'Memory' in data and type(data['Memory']) != str: 
                data['Memory'] = self.memory_string(data['Memory'])
            self.csv_data.append(data)

    def time_string(self, secs):
        """
        Helper function that creates time string from secs in form "SS.MS", "MM:SS.MS", or "HH:MM:SS.MS" based on the time
        
        Arguments
        ---------
            secs: float
                Time in seconds to convert to a string
        
        Returns
        -------
            time_str: string 
                A string with secs in the form "HH:MM:SS.MS", "MM:SS.MS", or "SS.MS" 
        """
        # Calculated number of hours, minutes, and seconds from secs
        hours = math.floor(secs // 60**2)
        minutes = math.floor((secs - (hours * 60**2)) / 60)
        seconds = secs - ((hours * 60**2) + (minutes * 60))
        
        # Build string, checking if hours and minutes are needed
        time_str = ""
        time_str += f"{hours:02}:" if secs >= 3600 else ""
        time_str += f"{minutes:02}:" if secs >= 60 else ""
        time_str += f"{seconds:05.2f}"
        time_str += "s" if secs < 60 else ""
        
        return time_str
    
    ### Allow 0 / negative mem datas to show no increase / decrease in mem usage
    def memory_string(self, mem): 
        """
        Helper function to create string from mem
        
        Arguments
        ---------
            mem: float
                memory value in bytes to convert to a string
        
        Returns
        -------
            mem_str: string 
                A string with memory converted to TB, GB, MB, KB, or B
        """
        # Dictionary containing units and their corresponding power of 1024
        UNITS = {4: "TB", 3: "GB", 2: "MB", 1: "KB", 0: "B"} 
        
        # Make sure mem has been given a value (incase called on a section of data that is undefined) 
        mem = mem if mem is not None and mem > 0 else 1
        
        # Determine best size units (most likely GB, MB, or KB)
        unit_power = math.floor(math.log(mem, 1024))
        unit = UNITS[unit_power]
        
        # Convert memory into prefered unit size
        mem_usage = mem / 1024**unit_power
        
        # Return string
        return f"{mem_usage:.4f} {unit}"
        
        
    def get_comparison(self, data):
        """
        Helper function to get value of a specific identifier in data
        
        Arguments
        ---------
            data: dictionary
                recorded data values with specific identifier and type (1D dictionary)
        
        Returns
        -------
            value: float
                The value of the comparison (or start value if end value does not exist)
        """
        # Get start from provided data and attempt to get end
        start = data['Start']
        end = data.get('End')
        
        # Return the difference over the interval of data (end-start)
        return end - start if end is not None else start
    
    def get_peak(self, func, type, data=None):
        """
        Helper function to get the peak (max or min) value of a specified type from stored data
        
        Arguments
        ---------
            func: function (max or min)
                function used to get the peak data
            type: string
                the data type to retrieve the peak from
            data: list of dictionaries (default None)
                allows use of this function on non-stored data values (must be 3D dictionary)
        Returns
        -------
            step: string
                the identifier where the peak value is located
            prox: string
                the proximity value (start or end) where the peak value is located
            peak: float
                the peak value of the specific metric
        """
        # Allow use of this function for non-stored data dictionaries (must be 3D dictionary)
        data = self.data if data is None else data
        
        # Converts data from {step: {type: {proximity: value}}} to {(step, type, proximity): value} (3D->1D) and filters all non applicable entries
        filtered_data = {(x, y, z): data[x][y][z] for x in data for y in data[x] for z in data[x][y] if y == type}
        
        # Gets index of peak value (as determined by func) in dictionary, then gets the peak value and returns all three together
        (step,_,prox) = func(filtered_data, key=filtered_data.get) if filtered_data != dict() else (None, None, None)
        peak = self.data[step][type][prox] if step != None else 0
        return step, prox, peak
    
    def get_time_elapsed(self, start=None, end=None): 
        """
        Helper function to get time elapsed from earliest saved time value to latest or specified time value (end)
        
        Arguments
        ---------
            start: string (default None)
                identifier of start point for getting ellapsed time
            end: string (default None) 
                identifier to find the time elapse to (time from start -> end)

        Returns
        -------
            start_step: string
                identifier where the earliest recored time is found
            end_step: string
                identifier where the latest recored time (or latest time before end) is found
            elapsed: float
                time elapsed between start and end
        """
        # Get keys from our data list
        keys = list(self.data.keys())
        
        # define start and end index of our time frame (using default or provided values) 
        start_idx = 0 if start == None else keys.index(start)
        end_idx = len(keys) if end == None else keys.index(end)
        
        # Create copy of self.data array with elements in our specified range
        data = {keys[x]: self.data[keys[x]] for x in range(start_idx, end_idx)}
        
        # Get the largest and smallest values in the specified interval
        start_step, _, start = self.get_peak(min, 'Time', data)
        end_step, _, finish = self.get_peak(max, 'Time', data)
        
        # Return results
        return start_step, end_step, finish - start 
    
    ### Can short by generalizing if statements
    def print_data(self, type, identifier, eol="\n", final=False): 
        """
        Prints the saved data
        
        Arguments
        ---------
            type: string
                the specific stored data type we want to print
            identifier: string
                the specific stored data instance we want to print
            eol: string (default "\n")
                the end of line character to use when spliting up lines
            final: boolean (default False)
                a boolean stating if this is the final print (meaning execution is complete)
        """
        
        # Define helper lists to make output easier
        STRING_FUNC = {'Time': self.time_string, 'Memory': self.memory_string}
        AFTER_STRING = {'Time': 'Elapsed', 'Memory': 'Used'}
        
        # Get start / end values of specified type from identifier
        start = self.data[identifier][type].get('Start')
        end = self.data[identifier][type].get('End')
            
        # If the identifier is 'Runtime' (the default value) and there is no end value yet, we do nothing 
        if identifier == 'Runtime' and end == None: 
            return
        
        # Prints a single data value (no comparison) (not useful for time data)
        if type != 'Time': 
            # Determine if we are printing before or after the event
            proximity = 'Before' if end == None else 'After'
            
            # Determine the value to print and make into string according to its type
            value = start if end == None else end
            value_str = STRING_FUNC[type](value)
            
            # Whether to add eol character (don't want eol if we have more to print)
            end_line = "" if end == None or self.log_level > 1 else eol
            
            # Create string to print according to determined values 
            data_str = f"{type} {proximity} {identifier}: {value_str}{end_line}"
            print(data_str) 
        
        # Prints comparison data (always for time, only for memory when log_level > 1)
        if end != None and (type == 'Time' or self.log_level > 1): 
            # Get comparison value (either time difference or memory difference) and convert to formatted string
            used_by_id = self.get_comparison(self.data[identifier][type])
            value = STRING_FUNC[type](used_by_id)
            
            # Add end of line if we have no more data to print 
            end_line = eol if self.log_level < 3 else ""
            
            # Create and print string
            proc_used = f"{type} {AFTER_STRING[type]} by {identifier}: {value}{end_line}"
            print(proc_used)
        
        # Prints total time elapsed by the program
        if end != None and self.log_level > 2 and type == 'Time': 
            # Get elasped time and convert to formatted string
            _, _, elapsed = self.get_time_elapsed()
            value = STRING_FUNC['Time'](elapsed)
            
            # Add formatting if this is the final datapoint
            start_line = "\n" if final else ""
            end_line = eol if not final else ""
            
            # Create and print string
            elap = f"{start_line}Total Time Elapsed: {value}{end_line}"
            print(elap)
        
        # Prints peak memory usage 
        if final and self.log_level > 2 and type == 'Memory': 
            # Get peak memory used and format as string 
            _, _, peak = self.get_peak(max, 'Memory')
            value = STRING_FUNC['Memory'](peak)
            
            # Create and print string
            peak_mem = f"\nPeak Memory Usage: {value}\n"
            print(peak_mem)
    
    def append_dict(self, type, value, identifier="Runtime", final=False): 
        """
        Appends data to data dictionary
        
        Arguments
        ---------
            type: string
                The data type being added to the dictionary (currently either 'Time' or 'Memory')
            value: float
                The value to be stored
            identifier: string (default "Runtime")
                The key to use when adding value to the dictionary. Considered the time step. Also used for labeling outputs
            final: boolean (default "True")
                Whether or not this is the final data entry
        """
        # Checks if the provided dictionary has an entry for identifier
        if identifier in self.data and type in self.data[identifier]: 
            # Update existing entry
            self.data[identifier][type]["End"] = value 
        elif identifier in self.data: 
            # Create new datatype entry for identifier dictionary
            self.data[identifier][type] = {"Start": value}
        else: 
            # Create new entry with Start value
            self.data[identifier] = {type: {"Start": value}}
                
        # If log_level > 0, print newly recorded data to console
        if self.log_level > 0:
            self.print_data(type, identifier, final=final)
            
    def build_csv_data(self): 
        """
        Creates CSV data using self.header and self.data and stores results in self.csv_data
        """
        for identifier in self.data:
            step = identifier
            time = None 
            mem = None 
            
            if 'Time' in self.data[identifier] and 'End' in self.data[identifier]['Time']: 
                time = self.time_string(self.get_comparison(self.data[identifier]['Time']))
            
            if 'Memory' in self.data[identifier]: 
                mem = self.memory_string(self.get_comparison(self.data[identifier]['Memory']))
            
            self.csv_data.append({'Step': step, 'Time': time, 'Memory': mem})
        
        if 'Time' in self.data: 
            time = self.time_string(self.get_time_elapsed()[2])
        
        if 'Memory' in self.data: 
            mem = self.memory_string(self.get_peak(max, 'Memory')[2])
        
        self.csv_data.append({'Step': 'Totals', 'Time': time, 'Memory': mem})

                  
    def write_csv(self, filename="data.csv"): 
        """
        Writes saved data to csv file
        
        Arguments 
        ---------
            filename: string (default "data.csv")
                Name of file to write csv to
        """
        # Convert filename to a .csv filename if it is not already
        root, ext = os.path.splitext(filename)
        if not ext or ext != '.csv': 
            ext = '.csv'
        filename = root + ext
        
        # Create csv_data if it is being handled internally
        if self.create_csv_data:
            self.build_csv_data()
        
        # Try to write to csv, and catch IOError
        try: 
            # Open file to write, no newline delimiter to keep from added unnecessary blank lines, and write all data
            with open(filename, 'w', newline="") as csvfile: 
                writer = csv.DictWriter(csvfile, fieldnames=self.header)
                writer.writeheader()
                for row in self.csv_data: 
                    writer.writerow(row)
        except IOError: 
            print("Error, Could Not Write CSV")
    
    ### Can condense by generalizing string creation (use 2 seperate lists to track time and mem then combine for writing)
    ### Do all file writing in a single file? 
    def summary(self, filename="summary.txt"):
        """
        Writes summary of all timing and memory data over console, as well as to a specified txt file
        
        Arguments
        ---------
            filename: string (default "summary.txt")
                file to print summary data to
        """
        # Convert filename to a .txt file if it is not already
        root, ext = os.path.splitext(filename)
        if not ext or ext != '.txt': 
            ext = '.txt'
        filename = root + '_Summary' + ext
            
        # Create copy lists containing all relevant data, filtered by type
        time_data = {x: self.data[x][y] for x in self.data for y in self.data[x] if y == 'Time'}
        mem_data = {x: self.data[x][y] for x in self.data for y in self.data[x] if y == 'Memory'}
        
        if self.log_level >= 1: 
            print("\n\n* SUMMARY *\nTIMES\n-----")
        
        # Create list with all summary data that will be written to console / file
        sum_data = ["TIMES", "-----"]
        
        # Print all timing data
        for identifier in time_data: 
            # Skip 'Runtime' data for now and output at end of time section
            if identifier == 'Runtime':
                continue
            
            # Get time value over current identifier and create formatted string
            value = self.time_string(self.get_comparison(time_data[identifier]))
            task_timing_summary = f"{identifier} Time: {value}"
            
            # Print to console with correct log level
            if self.log_level >= 1: 
                print(task_timing_summary)
            
            # Add data to summary 
            sum_data.append(task_timing_summary)
        
        # Print runtime data if available
        if 'Runtime' in time_data: 
            # Get string representation of total runtime
            value = self.time_string(self.get_comparison(time_data['Runtime']))
            task_timing_summary = f"Total Runtime: {value}"
            
            # Print to console with correct log level 
            if self.log_level >= 1: 
                print(task_timing_summary)
                print("\n")
             
            # Add data to summary
            sum_data.append(task_timing_summary)
            sum_data.append('\n')
        
        if self.log_level >= 1: 
            print("MEMORY\n------")
        
        # Start Memory section of summary 
        sum_data.extend(['MEMORY', '------'])
        
        # Print all memory data
        for identifier in mem_data: 
            # Skip 'Runtime' for later output
            if identifier == 'Runtime': 
                continue
            
            # Get all memory data and get memory usage data
            before = self.memory_string(mem_data[identifier]['Start'])
            after = self.memory_string(mem_data[identifier]['End'])
            val = self.memory_string(self.get_comparison(mem_data[identifier]))
            
            # Format data into strings
            before_str = f"{identifier} Memory Before: {before}"
            after_str = f"{identifier} Memory Usage After {after}"
            val_str = f"{identifier} Used {val}:"
            
            # Print with correct log_level
            if self.log_level >= 1: 
                print(before_str)
                print(after_str)
                print(val_str)
            
            # Add data to summary
            sum_data.extend([before_str, after_str, val_str])
        
        # Get peak memory usage formatted as string
        if mem_data != dict():
            peak = self.memory_string(self.get_peak(max, 'Memory')[2])
            peak_str = f"Peak Memory Usage: {peak}"
        
            # Print with correct log_level 
            if self.log_level >= 1: 
                print(peak_str)
                
            # Add data to summary 
            sum_data.append(peak_str)
        
        # Print summary list to text file
        textfile = open(filename, "w")
        for d in sum_data:
            textfile.write(d + "\n")

        textfile.close()
        
    def add_data(self, time=None, mem=None, data=None, identifier="Runtime", final=False, filename=None, directory="."):
        """ 
        Adds provided data to associated storage lists
        
        Arguments
        ---------
            time: float (default None)
                Value in seconds of the time stamp to save
            mem: float (default None)
                Value in bytes of memory usage to save
            data: dictionary (default None)
                csv data to add for late writing (will not be used if self.create_csv_data is true)
            identifier: string (default 'Runtime')
                string to use to identify this specific data addition (will be used for labeling and outputting)
            filename: string (default None)
                file to save csv data to upon completion of program
        """
        # Add mem data if provided
        if mem is not None: 
            self.append_dict('Memory', mem, identifier, final)
        
        # Add time data if provided 
        if time is not None: 
            self.append_dict('Time', time, identifier, final)
        
        # Add csv data if data is provided and it is not being created internally
        if self.create_csv_data is False and data is not None: 
            self.append_csv(data)

        # Output relevant files if this is the final data added and a filename has been provided
        if final and filename != None: 
            # Write csv file to directory/filename.csv
            self.write_csv(os.path.join(directory, filename))
            
            # Write directory/filename.txt file with all recorded data and prints summary to console with correct log_level
            self.summary(os.path.join(directory,filename))
            
            # Return total time elapsed and peak memory usage
            return self.time_string(self.get_time_elapsed()[2]), self.memory_string(self.get_peak(max, 'Memory')[2])