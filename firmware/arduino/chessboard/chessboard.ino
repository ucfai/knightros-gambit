// ================================
// START: Pin definitons
// ================================

// UART
#define RX2 16
#define TX2 17

// Electromagnet
#define ELECTROMAGNET 23

// Switches and buttons
#define X_AXIS_ENDSTOP_SWITCH 18
#define Y_AXIS_ENDSTOP_SWITCH 19
#define CHESS_TIMER_BUTTON 21

// X motor pins
#define X_MOTOR_MS1 13
#define X_MOTOR_MS2 12
#define X_MOTOR_DIR 27
#define X_MOTOR_STEP 14
int xMotor[5] = {X_MOTOR_STEP, X_MOTOR_DIR, X_MOTOR_MS1, X_MOTOR_MS2, X_AXIS_ENDSTOP_SWITCH};

// Y motor pins
#define Y_MOTOR_MS1 26
#define Y_MOTOR_MS2 25
#define Y_MOTOR_DIR 32
#define Y_MOTOR_STEP 33
int yMotor[5] = {Y_MOTOR_STEP, Y_MOTOR_DIR, Y_MOTOR_MS1, Y_MOTOR_MS2, Y_AXIS_ENDSTOP_SWITCH};

// Shared motor pins
#define MOTOR_RESET 4
#define MOTOR_SLEEP 2
#define MOTOR_ENABLE 15

// ================================
// END: Pin definitons
// ================================

// Distance definitions
#define MILLIMETERS_PER_SQUARE 63
#define STEPS_PER_MILLIMETER 5
#define HOME_CALIBRATION_OFFSET 100
float stepsPerSpace;
int currentX, currentY;

// Motor directions
#define UP 0 
#define DOWN 1
#define LEFT 0
#define RIGHT 1 

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
#define ENDSTOP_PIN 4

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

  pinMode(X_AXIS_ENDSTOP_SWITCH, INPUT);
  pinMode(Y_AXIS_ENDSTOP_SWITCH, INPUT);
  pinMode(CHESS_TIMER_BUTTON, INPUT);

  stepsPerSpace = MILLIMETERS_PER_SQUARE * STEPS_PER_MILLIMETER;

  Serial2.begin(115200, SERIAL_8N1, RX2, TX2);
}

void loop()
{
  

}
