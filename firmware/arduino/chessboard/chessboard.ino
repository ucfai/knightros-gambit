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
#define RX2 16
#define TX2 17

// Electromagnet
#define ELECTROMAGNET 23

// Switches and buttons
#define X_AXIS_ENDSTOP_SWITCH 33
#define Y_AXIS_ENDSTOP_SWITCH 32
#define CHESS_TIMER_BUTTON 4

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
  Serial2.begin(115200, SERIAL_8N1, RX2, TX2);
  resetEDPins();
}

int steps;

void loop()
{
  

}

// Function Declarations
void stepForwardX() {
  digitalWrite(MOTOR1_DIR,LOW);
  digitalWrite(MOTOR1_STEP_PIN,HIGH);
  delay(1);
  digitalWrite(MOTOR1_STEP_PIN,LOW);
}

void stepForwardY(){
  digitalWrite(MOTOR2_DIR,LOW);
  digitalWrite(MOTOR2_STEP_PIN,HIGH);
  delay(1);
  digitalWrite(MOTOR2_STEP_PIN,LOW);
}

void resetEDPins() {
  digitalWrite(MOTOR1_DIR,LOW);
  digitalWrite(MOTOR2_DIR,LOW);
  digitalWrite(MOTOR1_STEP_PIN, LOW);
  digitalWrite(MOTOR2_STEP_PIN, LOW);
  digitalWrite(MOTOR1_MS1,LOW);
  digitalWrite(MOTOR1_MS2,LOW);
  digitalWrite(MOTOR2_MS1,LOW);
  digitalWrite(MOTOR2_MS2,LOW);
  digitalWrite(MOTOR_ENABLE, HIGH);
}

void halfStep() {
  digitalWrite(MOTOR1_MS1,HIGH);
  digitalWrite(MOTOR1_MS2,LOW);
  digitalWrite(MOTOR2_MS1,HIGH);
  digitalWrite(MOTOR2_MS2,LOW);
}

void quarterStep() {
  digitalWrite(MOTOR1_MS1,LOW);
  digitalWrite(MOTOR1_MS2,HIGH);
  digitalWrite(MOTOR2_MS1,LOW);
  digitalWrite(MOTOR2_MS2,HIGH);
}

void eighthStep() {
  digitalWrite(MOTOR1_MS1,HIGH);
  digitalWrite(MOTOR1_MS2,HIGH);
  digitalWrite(MOTOR2_MS1,HIGH);
  digitalWrite(MOTOR2_MS2,HIGH);
}

// Message Parsing
bool parse_message_from_pi(char * buffer);

// Piece Movement
void move_straight(int startRow, int startCol, int endRow, int endCol);
void move_to_graveyard(int startRow, int startCol, char color, char type);
void castle(char color, char queenside_or_kingside);
void move_along_edges(int startRow, int startCol, int endRow, int endCol);