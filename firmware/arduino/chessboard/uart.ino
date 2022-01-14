char incomingByte;
char buffer[6];
int byteNum = -1; // -1 indicates that the start code hasn't been received
char currentState = '0';
char errorCode;
volatile unsigned long previous_activation_time = 0;
bool movementFlag = false; // Prevents END_TURN transmissions during movement function
volatile bool transmitFlag = false; // Queues END_TURN transmission after movement function

enum ArduinoState
{
    IDLE = '0',
    EXECUTING = '1',
    END_TURN = '2',
    ERROR = '3'
};

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

// Send message to Pi when the chess timer is pressed
void chessTimerISR()
{
    unsigned long current_time = millis();
    
    // Check if the difference between button presses is longer than the debounce time
    if (current_time - previous_activation_time > DEBOUNCE_TIME || 
       (current_time < previous_activation_time && previous_activation_time - current_time > DEBOUNCE_TIME))
    {
        previous_activation_time = current_time; 
        if (movementFlag)
        {
            transmitFlag = true;
        }
        else
        {
            sendMessageToPi(END_TURN, 0, buffer[5]);
        }
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
                sendMessageToPi(currentState, buffer[5], errorCode);
            }

            byteNum = 0;
        }
        // Add byte to buffer
        else if (byteNum  !=  -1)
        {
            buffer[byteNum++] = incomingByte;
        }

        // Check if the buffer is full, process input
        if (byteNum  ==  6)
        {
            // Reset buffer position
            byteNum = -1;

            // Process input
            // Returns true for valid input and false for invalid input.
            currentState = EXECUTING;
            if (validateMessageFromPi(buffer))
            { 
                // Sends acknowledgement
                sendMessageToPi(currentState, buffer[5], errorCode);

                movementFlag = true;
                makeMove(buffer);
                movementFlag = false;

                // Tell Pi that the chess timer button was pressed
                if (transmitFlag)
                    sendMessageToPi(END_TURN, 0, buffer[5]);
                transmitFlag = false;
            }

            // Sends move success/error
            sendMessageToPi(currentState, buffer[5], errorCode);
        }
    }
}

// Check that the instruction is valid
bool validateMessageFromPi(char * message)
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
    errorCode = NO_ERROR;
    return true;
}

bool makeMove(char * message)
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

void sendMessageToPi(char status, char moveCount, char errorMessage)
{
    Serial2.write('~');
    Serial2.write(status);
    Serial2.write(errorMessage);
    Serial2.write(moveCount);
}

bool isInvalidCoord (char c)
{
    return c < 'A' || c > 'L';
}
