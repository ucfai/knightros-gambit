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

#define MOTOR2_MS1 27
#define MOTOR2_MS2 2
#define MOTOR2_DIR 36
#define MOTOR2_STEP_PIN 36

// UART
#define SOFT_RX 16
#define SOFT_TX 17

// Electromagnet
#define MAGNET 23

// Switches and buttons
#define X_AXIS_ENDSTOP_SWITCH 33
#define Y_AXIS_ENDSTOP_SWITCH 32
#define CHESS_TIMER_BUTTON 4

void setup()
{
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
}

void loop()
{
  

}