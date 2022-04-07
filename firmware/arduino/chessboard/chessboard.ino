// ==========================================
// START: Constant definitons
// ==========================================

// Debug flag for enabling printing debug messages to the serial monitor
// This will have different debug levels for different groups of debug data
// (Note: Each level will print messages from the previous level(s) as well)

// Current Debug Levels:
// 0. This level is used to specify that no messages should be printed
// 1. UART_LEVEL:     This level will be used to print uart messages
// 2. FUNCTION_LEVEL: This level will be used to print information about importantt funciton calls 
#define DEBUG 1

enum DebugLevels
{
  UART_LEVEL = 1,
  FUNCTION_LEVEL = 2
};

enum UARTMisc
{
  RX2 = 16,
  TX2 = 17,
  INCOMING_MESSAGE_LENGTH = 6,
  OUTGOING_MESSAGE_LENGTH = 3
};

// 127 is for 50% duty cycle, 255 is for 100% duty cycle
enum ElectromagnetMisc
{
  ELECTROMAGNET = 23,
  PWM_FULL = 255,
  PWM_HALF = 127,
  EM_PWM_CHANNEL = 0,
  PWM_FREQUENCY = 100,
  PWM_RESOLUTION = 8
};

enum SwitchesMisc
{
  X_AXIS_MAX_ENDSTOP = 36,
  Y_AXIS_MAX_ENDSTOP = 34,
  X_AXIS_ZERO_ENDSTOP = 39,
  Y_AXIS_ZERO_ENDSTOP = 35,
  CHESS_TIMER_BUTTON = 21,
  DEBOUNCE_TIME = 100    
};

enum XMotorPins
{
  X_MOTOR_MS1 = 13,
  X_MOTOR_MS2 = 12,
  X_MOTOR_DIR = 27,
  X_MOTOR_STEP = 14
};
uint8_t xMotor[6] = {X_MOTOR_STEP, X_MOTOR_DIR, X_MOTOR_MS1, 
                     X_MOTOR_MS2, X_AXIS_MAX_ENDSTOP, X_AXIS_ZERO_ENDSTOP};

enum YMotorPins
{
  Y_MOTOR_MS1 = 26,
  Y_MOTOR_MS2 = 25,
  Y_MOTOR_DIR = 32,
  Y_MOTOR_STEP = 33
};
uint8_t yMotor[6] = {Y_MOTOR_STEP, Y_MOTOR_DIR, Y_MOTOR_MS1, 
                     Y_MOTOR_MS2, Y_AXIS_MAX_ENDSTOP, Y_AXIS_ZERO_ENDSTOP};
                  
enum MotorArrayIndicies
{
  STEP_PIN = 0,
  DIR_PIN = 1,
  MS1_PIN = 2,
  MS2_PIN = 3,
  ZERO_ENDSTOP_PIN = 4,
  MAX_ENDSTOP_PIN = 5
};

enum SharedMotorPins
{
  MOTOR_ENABLE = 4,
  MOTOR_RESET = 2,
  MOTOR_SLEEP = 15
};

enum ArduinoState
{
  IDLE = '0',
  EXECUTING = '1',
  END_TURN = '2',
  ERROR = '3'
};
volatile char currentState = IDLE;

enum DistanceConstants
{
  MILLIMETERS_PER_UNITSPACE = 32,
  STEPS_PER_MILLIMETER = 5,  // Whole steps per millimeter
  HOME_CALIBRATION_OFFSET = 100,
  TOTAL_UNITSPACES = 22
};  

// Sets direction of motor to move in the positive or negative direction regardless of axis
// Since the origin is at the bottom left corner, left/downward movement is considered negative (NEG_DIR), 
// and right/upward movement is positive (POS_DIR)
enum Direction
{
  POS_DIR = 0,
  NEG_DIR = 1
}; 

enum StepSize
{
  WHOLE_STEPS = 1,
  HALF_STEPS = 2,
  QUARTER_STEPS = 4,
  EIGHTH_STEPS = 8
};

enum CircleFunctionConsts
{
  // Number of concentric circles to make when centering pieces
  NUM_CIRCLES = 3,
  // Number of different slope settings for each quarter circle
  NUM_SLOPES_PER_QUARTER_CIRCLE = 9
};

// ==========================================
// START: Global Variable Declarations
// ==========================================

// UART input and flags
char moveCount;
volatile char extraByte;

// Create two message buffers, 1: incoming message and 2: last received message
// tempCharPtr is used to swap between the two message buffers
volatile char messageBuffer1[INCOMING_MESSAGE_LENGTH];
volatile char messageBuffer2[INCOMING_MESSAGE_LENGTH];
volatile char * tempCharPtr;
volatile char * rxBufferPtr = messageBuffer1;
volatile char * receivedMessagePtr = messageBuffer2;
volatile char sentMessage[OUTGOING_MESSAGE_LENGTH]; // holds status, extraByte, and moveCount (in that order)

// Flags are set asynchronously in uart.ino to begin processing their respective data
// When receivedMessageValidFlag == true, rxBufferPtr holds a complete and 
// unprocessed message from the pi
volatile bool receivedMessageValidFlag = false;
volatile bool buttonFlag = false;  // Queues END_TURN transmission when the chess timer is pressed
volatile bool uartMessageIncompleteFlag = false;  // Queues an error message when UART drops bytes

// Number of whole steps per unit space
uint8_t stepsPerUnitSpace;

// currPositionX and currPositionY measure distance from the origin 
// (bottom left corner of the board) in eighth steps
uint16_t currPositionX, currPositionY;

// Maximum position that currPositionX/Y may reach
uint16_t maxPosition;  

void setup()
{
  // Define our pinModes
  pinMode(MOTOR_RESET, OUTPUT);
  pinMode(MOTOR_SLEEP, OUTPUT);
  pinMode(MOTOR_ENABLE, OUTPUT);
  
  pinMode(X_MOTOR_MS1, OUTPUT);
  pinMode(X_MOTOR_MS2, OUTPUT);
  pinMode(X_MOTOR_DIR, OUTPUT);
  pinMode(X_MOTOR_STEP, OUTPUT);

  pinMode(Y_MOTOR_MS1, OUTPUT);
  pinMode(Y_MOTOR_MS2, OUTPUT);
  pinMode(Y_MOTOR_DIR, OUTPUT);
  pinMode(Y_MOTOR_STEP, OUTPUT);

  pinMode(ELECTROMAGNET, OUTPUT);

  pinMode(X_AXIS_MAX_ENDSTOP, INPUT);
  pinMode(Y_AXIS_MAX_ENDSTOP, INPUT);
  pinMode(X_AXIS_ZERO_ENDSTOP, INPUT);
  pinMode(Y_AXIS_ZERO_ENDSTOP, INPUT);
  pinMode(CHESS_TIMER_BUTTON, INPUT);

  // Defines the board's units being used
  stepsPerUnitSpace = MILLIMETERS_PER_UNITSPACE * STEPS_PER_MILLIMETER;

  // Max position in terms of eighth steps
  maxPosition = 8 * stepsPerUnitSpace * TOTAL_UNITSPACES;

  // Sets PWM constants and pin/channel assignments for ledcWrite use
  ledcSetup(EM_PWM_CHANNEL, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(ELECTROMAGNET, EM_PWM_CHANNEL);

  // Initializes global 2d array `pulsesPerSlope` which is used to define circle paths
  // that are used in the `makeCircle()` function in circleFunction.ino
  calculatePulsesPerSlope();

  Serial2.begin(9600, SERIAL_8N1, RX2, TX2);
  Serial.begin(115200);

  if (DEBUG >= UART_LEVEL)
  {
    Serial.println();
    Serial.println("Starting Program...");
  }
  
  attachInterrupt(digitalPinToInterrupt(CHESS_TIMER_BUTTON), chessTimerISR, RISING);
}

void loop()
{
  checkForInput();

  // Process the received message
  if (receivedMessageValidFlag)
  {
    receivedMessageValidFlag = false;

    currentState = EXECUTING;
    if (validateMessageFromPi(receivedMessagePtr))
    { 
      // Sends acknowledgement
      sendParamsToPi(currentState, extraByte, moveCount);
      makeMove(receivedMessagePtr);
    }

    // Sends move success/error
    // These variables can be changed inside the makeMove function
    sendParamsToPi(currentState, extraByte, moveCount);
  }

  // Transmit button press
  if (buttonFlag)
  {
    sendParamsToPi(currentState, extraByte, moveCount);
    buttonFlag = false;
  }

  // Transmit an error message if an incoming UART message is missing bytes
  if (uartMessageIncompleteFlag)
  {
    uartMessageIncompleteFlag = false;
    sendParamsToPi(currentState, extraByte, moveCount);
  }
}
