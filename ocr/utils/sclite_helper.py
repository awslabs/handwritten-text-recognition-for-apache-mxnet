import os
import subprocess
import re
import uuid

class ScliteHelper():
    '''
    The Sclite helper class calculates the word error rate (WER) and charater error rate (CER)
    given a predicted and actual text.
    This class uses sclite2.4 (ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.10-20151007-1312Z.tar.bz2)
    and formats the data according.
    Parameters
    ----------
    sclite_location: optional, default="sctk-2.4.10/bin"
        Location of the sclite_program
    tmp_file_location: optional, default=tmp
        folder to store the temporary text files.
    '''

    def __init__(self, sclite_location=os.path.join("..", "SCTK", "bin"),
                 tmp_file_location="tmp", use_uuid=True):
        # Check if sclite exists
        assert os.path.isdir(sclite_location), "{} does not exist".format(sclite_location)
        sclite_error = "{} doesn't contain sclite".format(sclite_location)
        retries = 10
        for i in range(retries):
            if self._test_sclite(sclite_location):
                break
            elif i == retries-1:
                raise sclite_error
        self.sclite_location = sclite_location 
        if use_uuid:
            tmp_file_location += "/" + str(uuid.uuid4())
        # Check if tmp_file_location exists
        if not os.path.isdir(tmp_file_location):
            os.makedirs(tmp_file_location)
        self.tmp_file_location = tmp_file_location
        self.predicted_text = []
        self.actual_text = []
        
    def clear(self):
        '''
        Clear the class for new calculations.
        '''
        self.predicted_text = []
        self.actual_text = []

    def _test_sclite(self, sclite_location):
        sclite_path = os.path.join(sclite_location, "sclite")
        command_line_options = [sclite_path]
        try:
            subprocess.check_output(command_line_options, stderr=subprocess.STDOUT)
        except OSError:
            return False
        except subprocess.CalledProcessError:
            return True
        return True

    def _write_string_to_sclite_file(self, sentences_arrays, filename):
        SPEAKER_LABEL = "(spk{}_{})"
        # Split string into sentences
        converted_string = ''
        for i, sentences_array in enumerate(sentences_arrays):
            for line, sentence in enumerate(sentences_array):
                converted_string += sentence + SPEAKER_LABEL.format(i+1, line+1) + "\n"
        
        # Write converted_string into file
        filepath = os.path.join(self.tmp_file_location, filename)
        with open(filepath, "w") as f:
            f.write(converted_string)

    def _run_sclite(self, predicted_filename, actual_filename, mode, output):
        '''
        Run command line for sclite.
        Parameters
        ---------
        predicted_filename: str
            file containing output string of the network  
        actual_filename: str
            file containing string of the label
        mode: string, Options = ["CER", "WER"]
            Choose between CER or WER
        output: string, Options = ["print", "string"]
            Choose between printing the output or returning a string 
        Returns
        -------
        
        stdoutput
            If string was chosen as the output option, this function will return a file 
            containing the stdout
        '''
        assert mode in ["CER", "WER"], "mode {} is not in ['CER', 'WER]".format(mode)
        assert output in ["print", "string"], "output {} is not in ['print', 'string']".format(
            output)

        command_line = [os.path.join(self.sclite_location, "sclite"),
                        "-h", os.path.join(self.tmp_file_location, predicted_filename),
                        "-r", os.path.join(self.tmp_file_location, actual_filename),
                        "-i", "rm"]
        if mode == "WER":
            pass # Word error rate is by default
        
        retries = 10
        
        for i in range(retries):
            try:
                if mode == "CER":
                    command_line.append("-c")
                if output == "print":
                    subprocess.call(command_line)
                elif output == "string":
                    cmd = subprocess.Popen(command_line, stdout=subprocess.PIPE)
                    return cmd.stdout
            except:
                print("There was an error")

    def _print_error_rate_summary(self, mode, predicted_filename="predicted.txt",
                                  actual_filename="actual.txt"):
        '''
        Print the error rate summary of sclite
        
        Parameters
        ----------
        mode: string, Options = ["CER", "WER"]
            Choose between CER or WER
        '''               
        self._run_sclite(predicted_filename, actual_filename, mode, output="print")

    def _get_error_rate(self, mode, predicted_filename="predicted.txt",
                                  actual_filename="actual.txt"):
        '''
        Get the error rate by analysing the output of sclite
        Parameters
        ----------
        mode: string, Options = ["CER", "WER"]
            Choose between CER or WER
        Returns
        -------
        number: int
           The number of characters or words depending on the mode selected. 
        error_rate: float
        '''
        number = None
        er = None
        output_file = self._run_sclite(predicted_filename, actual_filename,
                                       mode, output="string")

        match_tar = r'.*Mean.*\|.* (\d*.\d) .* (\d*.\d).* \|'
        for line in output_file.readlines():
            match = re.match(match_tar, line.decode('utf-8'), re.M|re.I)
            if match:
                number = match.group(1)
                er = match.group(2)
        assert number != None and er != None, "Error in parsing output."
        return float(number), 100.0 - float(er)
        
    def _make_sclite_files(self, predicted_filename="predicted.txt",
                           actual_filename="actual.txt"):
        '''
        Run command line for sclite.
        Parameters
        ---------
        predicted_filename: str, default: predicted.txt
            filename of the predicted file
        actual_filename: str, default: actual.txt
            filename of the actual file
        '''            
        self._write_string_to_sclite_file(self.predicted_text, filename=predicted_filename)
        self._write_string_to_sclite_file(self.actual_text, filename=actual_filename)
        
    def add_text(self, predicted_text, actual_text):
        '''
        Function to save predicted and actual text pairs in memory.
        Running the future fuctions will generate the required text files.
        '''
        self.predicted_text.append(predicted_text)
        self.actual_text.append(actual_text)

    def print_wer_summary(self):
        '''
        see _print_error_rate_summary for docstring
        '''
        self._make_sclite_files()
        self._print_error_rate_summary(mode="WER")
        
    def print_cer_summary(self):
        '''
        see _print_error_rate_summary for docstring
        '''
        self._make_sclite_files()
        self._print_error_rate_summary(mode="CER")

    def get_wer(self):
        '''
        See _get_error_rate for docstring
        '''
        self._make_sclite_files()
        return self._get_error_rate(mode="WER")
        
    def get_cer(self):
        '''
        See _get_error_rate for docstring
        '''
        self._make_sclite_files()
        return self._get_error_rate(mode="CER")

if __name__ == "__main__":
    cls = ScliteHelper()
    actual1 = 'Jonathan loves to eat apples. This is the second sentence.'
    predicted1 = 'Jonothon loves to eot. This is the second santense.'
    
    cls.add_text(predicted1, actual1)
    actual2 = 'Jonathan loves to eat apples. This is the second sentence.'
    predicted2 = 'Jonothan loves to eot. This is the second santense.'
    cls.add_text(predicted2, actual2)

    cls.print_cer_summary()
    num, er = cls.get_cer()
    print(num, er)