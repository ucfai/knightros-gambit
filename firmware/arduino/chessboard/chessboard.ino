// Pin definitions
// ================================
// Shared motor pins
#define MOTOR_RESET 6
#define MOTOR_SLEEP 13
#define MOTOR_ENABLE 7

// Separate motor pins
#define MOTOR1_MS1 14
#define MOTOR1_MS2 12
#define MOTOR1_DIR 34
#define MOTOR1_STEP_PIN 35
int xMotor[4] = {MOTOR1_STEP_PIN, MOTOR1_DIR, MOTOR1_MS1, MOTOR1_MS2};

#define MOTOR2_MS1 27
#define MOTOR2_MS2 2
#define MOTOR2_DIR 36
#define MOTOR2_STEP_PIN 36
int yMotor[4] = {MOTOR2_STEP_PIN, MOTOR2_DIR, MOTOR2_MS1, MOTOR2_MS2};

// Motor directions
#define FORWARD 0 
#define BACKWARD 1
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
#define STEPS_PER_SPACE 315 

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

  // Setup the UART here
  //Serial2.begin(115200, SERIAL_8N1, RX2, TX2);
}

void loop()
{
  

}
