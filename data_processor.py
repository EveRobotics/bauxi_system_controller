"""
The data processor interprets data received from the IPC manager.
"""

import threading
from lxml import etree
import Queue

import ipc_manager
from data_visualizer import DEG_PER_RAD

logger = ipc_manager.logger

TRUE_STRINGS = [ 'True', 'true', '1', 'yes' ]

class Configuration(object):
    
    class Port(object):
        """ For example:
        <!-- Port Configurations -->
        <port>
            <name>/dev/ttyACM0</name>
            <rate>115200</rate>
        </port>
        """
        
        def __init__(self, portNode):
            """ Port initializer.
            """
            self.__name = None
            self.__rate = None
            for child in portNode:
                if type(child) == etree._Comment: continue
                if child.tag == 'name':
                    self.__name = child.text
                if child.tag == 'rate':
                    self.__rate = int(child.text)
            return
        
        @property
        def name(self):
            return self.__name
    
        @property
        def rate(self):
            return self.__rate
        
    
    class Camera(object):
        """XML Parameter parser for the camera sub-system.
        """
        
        class Color(object):
            
            def __init__(self, hue, saturation, brightness
                         , threshHue, threshSat, threshBrt):
                
                self.__hue = hue
                self.__sat = saturation
                self.__brt = brightness
                self.__threshHue = threshHue
                self.__threshSat = threshSat
                self.__threshBrt = threshBrt
                return
            
            @property
            def hue(self):
                return self.__hue
            
            @property
            def sat(self):
                return self.__sat
            
            @property
            def brt(self):
                return self.__brt
            
            @property
            def threshHue(self):
                return self.__threshHue
            
            @property
            def threshSat(self):
                return self.__threshSat
            
            @property
            def threshBrt(self):
                return self.__threshBrt
            
        
        def __init__(self, cameraNode):
            """ Camera target color initializer.
            """
            self.__name = None
            self.__filterProperties = []
            for child in cameraNode:
                if type(child) == etree._Comment: continue
                elif child.tag == 'name':
                    self.__name = child.text
                elif child.tag == 'color':
                    for prop in child:
                        if type(prop) == etree._Comment: continue
                        elif prop.tag == 'hue':
                            hue = int(prop.text)
                        elif prop.tag == 'saturation':
                            saturation = int(prop.text)
                        elif prop.tag == 'brightness':
                            brightness = int(prop.text)
                        elif prop.tag == 'thresh_h':
                            threshHue = int(prop.text)
                        elif prop.tag == 'thresh_s':
                            threshSat = int(prop.text)
                        elif prop.tag == 'thresh_b':
                            threshBrt = int(prop.text)
                    color = self.Color(hue, saturation, brightness
                                      , threshHue, threshSat, threshBrt)
                    
                    self.__filterProperties.append(color)
            return
        
        @property
        def name(self):
            return self.__name
    
        @property
        def filterProperties(self):
            return self.__filterProperties
            
        
    
    class Compass(object):
        """Get the magnetic declanation from the XML configuration file.
        """
        
        def __init__(self, compassNode):
            """ Compass calibration initializer.
            """
            self.__properties = None
            for child in compassNode:
                if type(child) == etree._Comment: continue
                elif child.tag == 'declination':
                    self.__declination = float(child.text)
            return
        
        @property
        def declination(self):
            return self.__declination
        
    
    def __init__(self):
        """Open the configuration XML file and initialize the configuration 
        subsystem:
        Mission File: mission XML file name.
        Port List: A list of TTYs for serial data streaming.
        Camera: device name, target color properties.
        Compass: declination, calibration 
        Accelerometer:
        Gyroscope:
        """
        f = open('configuration.xml')
        fileText = f.read()
        f.close()
        root = etree.fromstring(fileText)
        
        self.__portList = []
        self.__camera = None
        self.__compass = None
        
        self.__enableIrRangefinders = True
        self.__enableGPS = True
        self.__missionFile = ''
        
        for child in root:
            if type(child) == etree._Comment: continue
            elif child.tag == 'port':
                port = self.Port(child)
                self.__portList.append(port) 
            elif child.tag == 'camera':
                self.__camera = self.Camera(child)
            elif child.tag == 'compass':
                self.__compass = self.Compass(child)
            elif child.tag == 'encoder_counts_per_meter':
                self.__encoderCountsPerMeter = float(child.text)
            elif child.tag == 'enable_ir_rangefinders':
                result = child.text in TRUE_STRINGS
                self.__enableIrRangefinders = result
            elif child.tag == 'enable_gps':
                result = child.text in TRUE_STRINGS
                self.__enableGPS = result
            elif child.tag == 'mission':
                self.__missionFile = child.text
        return
    
    
    @property
    def portList(self):
        return self.__portList
    
    @property
    def camera(self):
        return self.__camera
    
    @property
    def compass(self):
        """Fields:
        declination - magnetic declination
        """
        return self.__compass
        
    @property
    def encoderCountsPerMeter(self):
        """float."""
        return self.__encoderCountsPerMeter
    
    @property
    def enableIrRangefinders(self):
        """Boolean."""
        return self.__enableIrRangefinders
    
    @property
    def enableGPS(self):
        """Boolean."""
        return self.__enableGPS
    
    @property
    def missionFile(self):
        """String, GPX file name, file contains the Lat/Lon route."""
        return self.__missionFile
    


###############################################################################
# DATA PROCESSOR: base class.
###############################################################################

class DataProcessor(object):
    """Enque and dequeue messages for the system controller.
    """
    
    # Set to false to exit message processing threads.
    __runService = True
    
    # Static, list of IPC managers
    __ipcManagers = []
    __processingLoopEntered = False

    # Configuration information.
    _configuration = Configuration()
    
    @property
    def gpsDataChanged(self):
        """Flag to indicate the GPS data has changed to conditionally update
        the GPS location, in the mission planner."""
        return self._gpsDataChanged
    
    @property
    def encoderCountsPerMeter(self):
        """Get the number of encoder counts per meter."""
        return self._configuration.encoderCountsPerMeter
    
    @property
    def missionFile(self):
        """GPX file name, file contains lat/lon route information."""
        return self._configuration.missionFile
    
    
    def __init__(self, parent=True, missionPlanner=None):
        """ Data processor constructor.
        Subsystem initialization
        """ 
        if not parent:
            # Initialize some common fields used between the parent and the 
            # children, but then return immediately.
            self._txQueue = Queue.Queue()
            self._messagesProcessedCount = 0;
            return
        
        logger.info("Starting IPC managers.")
        # TODO: initialize the camera manager, and get the configurations
        # from the configuration XML file.
        for port in self._configuration.portList:
            portName = port.name
            baudRate = port.rate
            self.__ipcManagers.append(ipc_manager.IpcManager(portName, baudRate))
        
        # Setup the camera IPC manager.
        cameraName = self._configuration.camera.name
        filterProperties = self._configuration.camera.filterProperties
        cameraManager = ipc_manager.IpcManager(cameraName, filterProperties)
        self.__ipcManagers.append(cameraManager)
        
        # Initialize new processors here:
        # Data processors interpret the raw (minimally processed) sensor data.
        self.__inertialNav = InertialDataProcessor(missionPlanner)
        self.__spatialNav = SpatialDataProcessor(missionPlanner)
        self.__gpsNav = GpsDataProcessor(missionPlanner)
        self.__camera = CameraDataProcessor(missionPlanner)
        
        self.__processors = (self.__inertialNav, self.__spatialNav
                             , self.__gpsNav, self.__camera)
        
        # Message processor callbacks, used when receiving an IPC message.
        self.__processorTable = { 'i' : self.__inertialNav.processMessage
                                , 's' : self.__spatialNav.processMessage
                                , 'g' : self.__gpsNav.processMessage
                                , 'c' : self.__camera.processMessage }

        # Enqueue a message to transmit to a peripheral controller.
        self.__senderTable = { 'i' : self.__inertialNav.enqueueMessage
                             , 's' : self.__spatialNav.enqueueMessage
                             , 'g' : self.__gpsNav.enqueueMessage
                             , 'c' : self.__camera.enqueueMessage }
        
        t = threading.Thread(name='data_processor', target=self.runDataProcessors)
        # Check that the data processors have all started without issue.
        t.start()
        return
    
    
    def runDataProcessors(self):
        """ Poll the message loop for incoming IPC messages.
        """
        logger.info("Starting data processor.")
        __processingLoopEntered = True
        while self.__runService:
            for m in self.__ipcManagers:
                if not self.__runService:
                    logger.info("Stop requested, data processor run service = False")
                    break
                # Check for None message type and skip process step.
                message = m.getMessage()
                if message is not None:
                    self.processMessage(message)
                # Update state in the mission planner.
        
        for p in self.__processors:
            message = str(p) + ' messages processed: ' \
                + str(p._messagesProcessedCount)
            logger.info(message)
        logger.info("Stopped data processor.")
        return
    
    
    def processMessage(self, message):
        """
        """
        # Dispatch message to the relevant class's processor:
        if message is not None and message.Id is not None:
            self.__processorTable.get(message.Id, None)(message)
        return
    
    
    def shutdown(self):
        # Stop all child threads and the parent thread.
        for mgr in self.__ipcManagers:
            mgr.shutdown()
        self.__runService = False
        return
    
    
    # IPC message queue. System controller places messages in the
    # queue to be sent to the various controllers (outgoing queue).
    def enqueueMessage(self, message):
        """ Put a message in a message queue to be read by a specific 
        data processor and transmitted to a peripheral controller via IPC.
        """
        self._txQueue.put(message)
        return
    
    
    def sendMessage(self, message):
        """ Put a message in a message queue to be read by a specific 
        data processor and transmitted to a peripheral controller via IPC.
        """
        callback = self.__senderTable.get(message[0], None)
        if callback is not None:
            callback(message)
        return
    
    

class InertialDataProcessor(DataProcessor):
    """
    Process sensor data messages in the context of a class that 'understands'
    the data and how to represent it to the mission_planner and visually via 
    some graphical display (data visualization rather than text data display).
    """
    
    
    def __init__(self, missionPlanner):
        DataProcessor.__init__(self, False)
        self.__missionPlanner = missionPlanner
        
        self.__declination = self._configuration.compass.declination
        self.__message = None
        return
    
    
    def processMessage(self, message):
        # Dispatch message to the relevant class's processor:
        
        self.__message = message
        yaw = 360.0 - message.heading + self.__declination
        
        if yaw >= 360.0:
            yaw = yaw - 360.0
        elif yaw < 0.0:
            yaw = yaw + 360.0
        
        yawRadians = yaw / DEG_PER_RAD
        
        #print yawRadians, yawDegrees
        self._messagesProcessedCount += 1
        self.__missionPlanner.updateInertial(message, yawRadians, yaw)
        logger.debug(message.toString())
        return
    
    

class SpatialDataProcessor(DataProcessor):
    """Convert the counts to distances etc.."""

    ID_SPACIAL = 's'
    
    IR_DISTANCE_NOMINAL = 55
    
    def __init__(self, missionPlanner):
        DataProcessor.__init__(self, False)
        self.__missionPlanner = missionPlanner
        return
    
    
    def processMessage(self, message):
        """Update mission planner state ..."""
        # Possibly convert all distance sensor measurements to points in R3
        # using sensor pointing information from the configuration file.
        self._messagesProcessedCount += 1
        logger.debug(message.toString())
        
        if not self._configuration.enableIrRangefinders:
            # Doctor up the message to set IR values to nominal values.
            message.overrideIrValues(self.IR_DISTANCE_NOMINAL
                                     , self.IR_DISTANCE_NOMINAL
                                     , self.IR_DISTANCE_NOMINAL)
        
        self.__missionPlanner.updateSpacial(message)
        # Check the outgoing message queue for spacial messages.
        while not self._txQueue.empty():
            # TODO: peek in the queue to see that it is for us...
            txMessage = self._txQueue.get()
            if txMessage[0] == self.ID_SPACIAL:
                #print 'Spacial Data Processor: transmit message:', txMessage
                message.txCallback(txMessage[2:])
            
        return
    
    

class GpsDataProcessor(DataProcessor):
    
    def __init__(self, missionPlanner):
        DataProcessor.__init__(self, False)
        self.__missionPlanner = missionPlanner
        return
    
    
    def processMessage(self, message):
        """Process GPS messages.
        """
        if message.fields.Id == ipc_manager.GpsMessage.GPS_GGA \
                or message.fields.Id == ipc_manager.GpsMessage.GN_GGA:
            
            self.__processGgaMessage(message)
        
        logger.debug(message.toString())
        return
    
    
    def __processGgaMessage(self, message):
        """Process GGA messages. Compute northing and easting from lat/lon.
        """
        lat = convertDegMinToDeg(message.fields.latDegrees
                , message.fields.latMinutes, message.fields.nS)
        
        lon = convertDegMinToDeg(message.fields.lonDegrees
                , message.fields.lonMinutes, message.fields.eW)
        
        self._messagesProcessedCount += 1
        if lat != self.__missionPlanner.lat or lon != self.__missionPlanner.lon:
            if self._configuration.enableGPS:
                self.__missionPlanner.updateGps(message, lat, lon)
            
            self._gpsDataChanged = True
        else:
            self._gpsDataChanged = False
        
        return
    
    

class CameraDataProcessor(DataProcessor):
    """ Camera data processor class.
    
    Implements:
        processMessage: 
            Update code in the mission planner, send any messages in the TX 
            queue to the peripheral controller.
    Info: Data processor ID = 'c'
    """
    
    ID_CAMERA = 'c'
    
    def __init__(self, missionPlanner):
        """Constructor"""
        DataProcessor.__init__(self, False)
        self.__missionPlanner = missionPlanner
        return
    
    
    def processMessage(self, message):
        """Process message function, call into mission planner and update state.
        """
        self._messagesProcessedCount += 1
        self.__missionPlanner.updateVision(message)
        logger.debug(message.toString())
        if self._txQueue.qsize() > 0:
            txMessage = self._txQueue.get()
            if txMessage[0] == self.ID_CAMERA:
                logger.debug('Camera Data Processor: transmit message:' + txMessage)
                message.txCallback(txMessage[2:])
        return
    
    

# Some helper functions to convert between DD MM.MM to DD.DD..
# I.e. degrees minutes to degrees decimal degrees.
def convertDegMinToDeg(degrees, minutes, direction):
    degrees = degrees + minutes / 60.0
    if direction == 'S' or direction == 'W':
        degrees *= -1
    return degrees

# I.e. degrees minutes seconds to degrees decimal degrees.
def convertDegMinSecToDeg(degrees, minutes, seconds, direction):
    degrees = degrees + minutes / 60.0 + seconds / 3600
    if direction == 'S' or direction == 'W':
        degrees *= -1
    return degrees

