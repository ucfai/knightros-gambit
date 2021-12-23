// Pin definitions
// ================================
// Shared motor pins
#define MOTOR_RESET 6
#define MOTOR_SLEEP 13
#define MOTOR_ENABLE 7

// Motor directions
#define UP 0 
#define DOWN 1
#define LEFT 0
#define RIGHT 1 

// UART
#define RX2 16
#define TX2 17

// Electromagnet
#define ELECTROMAGNET 23

// Switches and buttons
#define X_AXIS_ENDSTOP_SWITCH 33
#define Y_AXIS_ENDSTOP_SWITCH 32
#define CHESS_TIMER_BUTTON 4

// Distance Definitions
#define MILLIMETERS_PER_SQUARE 63
#define STEPS_PER_MILLIMETER 5
#define HOME_OFFSET 100
float stepsPerSpace;
int currentX, currentY;

// Separate motor pins
#define MOTOR1_MS1 14
#define MOTOR1_MS2 12
#define MOTOR1_DIR 34
#define MOTOR1_STEP_PIN 35
int xMotor[5] = {MOTOR1_STEP_PIN, MOTOR1_DIR, MOTOR1_MS1, MOTOR1_MS2, X_AXIS_ENDSTOP_SWITCH};

#define MOTOR2_MS1 27
#define MOTOR2_MS2 2
#define MOTOR2_DIR 36
#define MOTOR2_STEP_PIN 36
int yMotor[5] = {MOTOR2_STEP_PIN, MOTOR2_DIR, MOTOR2_MS1, MOTOR2_MS2, Y_AXIS_ENDSTOP_SWITCH};

#define WHOLE_STEPS 1
#define HALF_STEPS 2
#define QUARTER_STEPS 4
#define EIGHTH_STEPS 8

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
  
  pinMode(MOTOR1_MS1, OUTPUT);
  pinMode(MOTOR1_MS2, OUTPUT);
  pinMode(MOTOR1_DIR, OUTPUT);
  pinMode(MOTOR1_STEP_PIN, OUTPUT);

  pinMode(MOTOR2_MS1, OUTPUT);
  pinMode(MOTOR2_MS2, OUTPUT);
  pinMode(MOTOR2_DIR, OUTPUT);
  pinMode(MOTOR2_STEP_PIN, OUTPUT);

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
