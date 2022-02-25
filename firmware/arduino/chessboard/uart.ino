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
                errorCode = INCOMPLETE_INSTRUCTION;
                errorFlag = true;
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
    if (message[0] == DIRECT || message[0] == EDGES)
    {
        if (isInvalidCoord(message[1]) || isInvalidCoord(message[2]) ||
            isInvalidCoord(message[3]) || isInvalidCoord(message[4]))
        {
            errorCode = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }
    }
    else if (message[0] == ALIGN)
    {
        if (isInvalidCoord(message[1]) || isInvalidCoord(message[2]))
        {
            errorCode = INVALID_LOCATION;
            currentState = ERROR;
            return false;
        }
    }
    else
    {
        // Invalid opcode
        errorCode = INVALID_OP;
        currentState = ERROR;
        return false;
    }

    // Update last valid move count
    moveCount = message[5];

    errorCode = NO_ERROR;
    return true;
}

bool makeMove(volatile char * message)
{
    // Move type 0
    if (message[0]  ==  DIRECT)
    {
        if (!moveDirect(message[2] - 'A', message[1] - 'A', message[4] - 'A', message[3] - 'A'))
        {
            currentState = ERROR;
            errorCode = MOVEMENT_ERROR;
            return false;
        }
    }
    // Move type 1
    else if (message[0] == EDGES)
    {
        if(!moveAlongEdges(message[2] - 'A', message[1] - 'A', message[4] - 'A', message[3] - 'A'))
        {
            currentState = ERROR;
            errorCode = MOVEMENT_ERROR;
            return false;
        }
    }
    // Move type 2
    else if (message[0] == ALIGN)
    {
        if(!alignPiece(message[2] - 'A', message[1] - 'A'))
        {
            currentState = ERROR;
            errorCode = MOVEMENT_ERROR;
            return false;
        }
    }
    else
    {
        // Invalid opcode
        errorCode = INVALID_OP;
        currentState = ERROR;
        return false;
    }

    // Move is valid and was made
    errorCode = NO_ERROR;
    currentState = IDLE;
    return true;
}

void sendMessageToPi(volatile char status, char moveCount, char errorMessage)
{
    Serial2.write('~');
    Serial2.write(status);
    Serial2.write(errorMessage);
    Serial2.write(moveCount);
}

bool isInvalidCoord (volatile char c)
{
    return c < 'A' || c > 'X';
}
