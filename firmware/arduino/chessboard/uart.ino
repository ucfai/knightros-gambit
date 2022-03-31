char incomingByte;
volatile unsigned long previousActivationTime = 0;

int byteNum = -1; // -1 indicates that the start code hasn't been received

// Send message to Pi when the chess timer is pressed
void chessTimerISR()
{
    unsigned long current_time = millis();
    
    // Check if the difference between button presses is longer than the debounce time
    if (current_time - previousActivationTime > DEBOUNCE_TIME || 
       (current_time < previousActivationTime && previousActivationTime - current_time > DEBOUNCE_TIME))
    {
        previousActivationTime = current_time;
        buttonFlag = true;
    }
}

// Wait for input
void serialEvent2()
{
    // Loop through all available bytes
    while (Serial2.available())
    {
        // Get byte
        incomingByte = (char) Serial2.read();

        // Reset buffer position
        if (incomingByte  ==  '~')
        {
            // Send message to Pi if the previous instruction was incomplete
            if (byteNum != -1)
            {
                currentState = ERROR;
                extraByte = INCOMPLETE_INSTRUCTION;
                uartMessageIncompleteFlag = true;
            }

            byteNum = 0;
        }
        // Add byte to buffer
        else if (byteNum  !=  -1)
        {
            rxBufferPtr[byteNum++] = incomingByte;
        }

        // Check if the buffer is full, process input
        if (byteNum  ==  INCOMING_MESSAGE_LENGTH)
        {
            // Reset buffer position
            byteNum = -1;

            // Swap rxBufferPtr and receivedMessagePtr pointers
            tempCharPtr = rxBufferPtr;
            rxBufferPtr = receivedMessagePtr;
            receivedMessagePtr = tempCharPtr;

            // Tell game loop to process input
            receivedMessageValidFlag = true;
        }
    }
}

// Check that the instruction is valid
bool validateMessageFromPi(volatile char * message)
{
    // If no error occurs to change the extraByte, it should store the opcode
    extraByte = message[0];

    if (message[0] == DIRECT || message[0] == EDGES)
    {
        if (isInvalidCoord(message[1]) || isInvalidCoord(message[2]) ||
            isInvalidCoord(message[3]) || isInvalidCoord(message[4]))
        {
            extraByte = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }
    }
    else if (message[0] == ALIGN)
    {
        if (isInvalidCoord(message[1]) || isInvalidCoord(message[2]))
        {
            extraByte = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }
    }
    else if (message[0] == INSTRUCTION)
    {
        // Check if message[5] holds an invalid instruction type
        if ((message[1] != ALIGN_AXIS  || message[2] < '0' || message[2] > '3') && 
            (message[1] != SET_ELECTROMAGNET || message[2] < '0' || message[2] > '1') &&
             message[1] != RETRANSMIT)
        {
            extraByte = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }
    }
    else
    {
        // Invalid opcode
        extraByte = INVALID_OP;
        currentState = ERROR;
        return false;
    }

    // Update last valid move count
    moveCount = message[5];

    return true;
}

bool makeMove(volatile char * message)
{
    // If no error occurs to change the extraByte, it should store the opcode
    extraByte = message[0];

    // Move type 0
    if (message[0]  ==  DIRECT)
    {
        // Since we're moving a piece, we want the magnet on, so pass in true.
        if (!moveDirect(message[2] - 'A', message[1] - 'A', message[4] - 'A', message[3] - 'A', true))
        {
            currentState = ERROR;
            extraByte = MOVEMENT_ERROR;
            return false;
        }
    }
    // Move type 1
    else if (message[0] == EDGES)
    {
        if(!moveAlongEdges(message[2] - 'A', message[1] - 'A', message[4] - 'A', message[3] - 'A'))
        {
            currentState = ERROR;
            extraByte = MOVEMENT_ERROR;
            return false;
        }
    }
    // Move type 2
    else if (message[0] == ALIGN)
    {
        if(!alignPiece(message[2] - 'A', message[1] - 'A'))
        {
            currentState = ERROR;
            extraByte = MOVEMENT_ERROR;
            return false;
        }
    }
    // Move type 3 - special instructions
    else if (message[0] == INSTRUCTION)
    {
        // Align Axis
        if (message[1] == ALIGN)
        {
          if (message[2] == '0')
            alignAxis(xMotor, ZERO_POSITION);
          else if (message[2] == '1')
            alignAxis(yMotor, ZERO_POSITION);
          else if (message[2] == '2')
            alignAxis(xMotor, MAX_POSITION);
          else if (message[2] == '3')
            alignAxis(yMotor, MAX_POSITION);
        }

        // Enable/Disable Electromagnet
        else if (message[1] == SET_ELECTROMAGNET)
        {
          if (message[2] == '0')
            digitalWrite(ELECTROMAGNET, LOW);
          else if (message[2] == '1')
            ledcWrite(EM_PWM_CHANNEL, PWM_HALF);
        }

        // Retransmit last message
        else if (message[1] == RETRANSMIT)
        {
            sendMessageToPi(sentMessage);
        }
    }
    else
    {
        // Invalid opcode
        extraByte = INVALID_OP;
        currentState = ERROR;
        return false;
    }

    // Move was made
    currentState = IDLE;
    return true;
}

void sendMessageToPi(volatile char * message)
{
    Serial2.write('~');
    Serial2.write(message[0]);
    Serial2.write(message[1]);
    Serial2.write(message[2]);
}

bool isInvalidCoord (volatile char c)
{
    return c < 'A' || c > 'W';
}
