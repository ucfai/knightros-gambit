// ================================
// START: Pin definitons
// ================================

// UART
#define RX2 16
#define TX2 17
#define INCOMING_MESSAGE_LENGTH 6

// UART input and flags
char moveCount;
volatile char errorCode;

// Create two message buffers, 1: incoming message and 2: last received message
// tempCharPtr is used to swap between the two message buffers
volatile char messageBuffer1[INCOMING_MESSAGE_LENGTH];
volatile char messageBuffer2[INCOMING_MESSAGE_LENGTH];
volatile char * tempCharPtr;
volatile char * rxBufferPtr = messageBuffer1;
volatile char * receivedMessagePtr = messageBuffer2;

// Flags are set asynchronously in uart.ino to begin processing their respective data
// When receivedMessageValidFlag == true, rxBufferPtr holds a complete and unprocessed message from Pi
volatile bool receivedMessageValidFlag = false;
volatile bool buttonFlag = false;  // Queues END_TURN transmission when the chess timer is pressed
volatile bool uartMessageIncompleteFlag = false;  // Queues an error message when UART drops bytes

enum ArduinoState
{
  IDLE = '0',
  EXECUTING = '1',
  END_TURN = '2',
  ERROR = '3'
};
volatile char currentState = IDLE;

enum MoveCommandType
{
  DIRECT = '0',
  EDGES = '1',
  ALIGN = '2'
};

enum ErrorCode
{
  NO_ERROR = '0',
  INVALID_OP = '1',
  INVALID_LOCATION = '2',
  INCOMPLETE_INSTRUCTION = '3',
  MOVEMENT_ERROR = '4'
};

// Electromagnet
#define ELECTROMAGNET 23
#define PWM_FULL 255
#define PWM_HALF 127 // 127 is for 50% duty cycle, 255 is for 100% duty cycle

// PWM setup for EM
#define PWM_FREQUENCY 100
#define EM_PWM_CHANNEL 0
#define PWM_RESOLUTION 8

// Switches and buttons
#define X_AXIS_MAX_ENDSTOP 36
#define Y_AXIS_MAX_ENDSTOP 34
#define X_AXIS_ZERO_ENDSTOP 39
#define Y_AXIS_ZERO_ENDSTOP 35
#define CHESS_TIMER_BUTTON 21

// X motor pins
#define X_MOTOR_MS1 13
#define X_MOTOR_MS2 12
#define X_MOTOR_DIR 27
#define X_MOTOR_STEP 14
uint8_t xMotor[6] = {X_MOTOR_STEP, X_MOTOR_DIR, X_MOTOR_MS1, X_MOTOR_MS2, X_AXIS_MAX_ENDSTOP, X_AXIS_ZERO_ENDSTOP};

// Y motor pins
#define Y_MOTOR_MS1 26
#define Y_MOTOR_MS2 25
#define Y_MOTOR_DIR 32
#define Y_MOTOR_STEP 33
uint8_t yMotor[6] = {Y_MOTOR_STEP, Y_MOTOR_DIR, Y_MOTOR_MS1, Y_MOTOR_MS2, Y_AXIS_MAX_ENDSTOP, Y_AXIS_ZERO_ENDSTOP};

// Shared motor pins
#define MOTOR_ENABLE 4
#define MOTOR_RESET 2
#define MOTOR_SLEEP 15

// ================================
// END: Pin definitons
// ================================

// Distance definitions
#define MILLIMETERS_PER_UNITSPACE 32
#define STEPS_PER_MILLIMETER 5  // Whole steps per millimeter
#define HOME_CALIBRATION_OFFSET 100
#define TOTAL_UNITSPACES 24

// Number of whole steps per unit space
int stepsPerUnitSpace;

// currPositionX and currPositionY measure distance from the origin (bottom left corner of the board) in eighth steps
int currPositionX, currPositionY;

// Maximum position that currPositionX/Y may reach
int maxPosition;

// Sets direction of motor to move in the positive or negative direction regardless of axis
// Since the origin is at the bottom left corner, left/downward movement is considered negative (NEG_DIR),
// and right/upward movement is positive (POS_DIR)
#define POS_DIR 0 
#define NEG_DIR 1

// Step size definitions
#define WHOLE_STEPS 1
#define HALF_STEPS 2
#define QUARTER_STEPS 4
#define EIGHTH_STEPS 8

// Motor array index definitions
#define STEP_PIN 0
#define DIR_PIN 1
#define MS1_PIN 2
#define MS2_PIN 3
#define ZERO_ENDSTOP_PIN 4
#define MAX_ENDSTOP_PIN 5

// Button debounce time (in milliseconds)
#define DEBOUNCE_TIME 100 

// Number of concentric circles to make when centering pieces
#define NUM_CIRCLES 3

// Number of different slope settings for each quarter circle
#define NUM_SLOPES_PER_QUARTER_CIRCLE 9

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

  // Homes the EM to the origin and sets currPositionX/Y to 0
  home();

  // Initializes global 2d array `pulsesPerSlope` which is used to define circle paths
  // that are used in the `makeCircle()` function in circleFunction.ino
  calculatePulsesPerSlope();

  Serial2.begin(115200, SERIAL_8N1, RX2, TX2);
  
  attachInterrupt(digitalPinToInterrupt(CHESS_TIMER_BUTTON), chessTimerISR, RISING);
}

void loop()
{
  // Process the received message
  if (receivedMessageValidFlag)
  {
    receivedMessageValidFlag = false;

    currentState = EXECUTING;
    if (validateMessageFromPi(receivedMessagePtr))
    { 
        // Sends acknowledgement
        sendMessageToPi(currentState, moveCount, errorCode);

        makeMove(receivedMessagePtr);
    }
    // Sends move success/error
    sendMessageToPi(currentState, moveCount, errorCode);
  }

  // Transmit button press
  if (buttonFlag)
  {
    sendMessageToPi(END_TURN, moveCount, 0);
    buttonFlag = false;
  }

  // Transmit an error message if an incoming UART message is missing bytes
  if (uartMessageIncompleteFlag)
  {
    uartMessageIncompleteFlag = false;
    sendMessageToPi(ERROR, moveCount, errorCode);
  }
}
