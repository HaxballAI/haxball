import os
import pickle

class DataHandler:
    def __init__(self, file_name):
        self.buffer = []
        self.file_name = file_name
        # Used for getting states from buffer
        self.line_considered = 0

    def loadIntoBuffer(self, content):
        self.buffer.append(content)

    def dumpBufferToFile(self):
        # Dump the content of the buffer to the end of file_name, if file doesn't
        # exist, create it.
        file_content = []
        if os.path.exists(self.file_name) and os.path.getsize(self.file_name) > 0:
            with open(self.file_name, 'rb') as rfp:
                file_content = pickle.load(rfp)
        file_content.append(self.buffer)

        with open(self.file_name, 'wb') as wfp:
            pickle.dump(file_content, wfp)

        # Reset the buffer since the content of the buffer was dumped
        self.resetBuffer()

    def loadFileToBuffer(self):
        self.resetBuffer()
        if os.path.exists(self.file_name) and os.path.getsize(self.file_name) > 0:
            with open(self.file_name, 'rb') as rfp:
                self.buffer = pickle.load(rfp)

    def getNextStateFromBuffer(self):
        # Returns the next state from the buffer, returns None if out of Range
        if self.line_considered >= self.buffer.size() or self.line_considered < 0:
            return None
        else:
            self.line_considered = 0
            return self.buffer[self.line_considered]

    def printBuffer(self):
        # Used for debugging purposes, resets the buffer in the process
        print(len(self.buffer))
        for element in self.buffer:
            print(element)


    def resetBuffer(self):
        self.buffer = []
        self.line_considered = 0

    def getBuffer(self):
        return self.buffer

    def getFileContent(self):
        self.loadFileToBuffer()
        return self.getBuffer()
