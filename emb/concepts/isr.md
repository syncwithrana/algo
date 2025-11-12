**Q:** What is an Interrupt?
- An interrupt is a signal to the processor that something needs immediate attention.
- It temporarily stops the current program, saves its state, and jumps to a specific function (ISR) to handle the event.


**Q:** What is an ISR?
- An Interrupt Service Routine (ISR) is a special function that runs when an interrupt occurs.
- It performs minimal, fast operations to handle that event (like reading data, clearing flags, etc.), then returns control to the main program.


**Q:** What happens in CPU when an interrupt occurs?
- CPU saves current context (program counter, registers).
- CPU disables further interrupts (optional or automatic depending on architecture).
- CPU jumps to ISR (using vector table).
- ISR runs and clears the interrupt flag.
- CPU restores context and resumes previous execution.


**Q:** Difference between polling and interrupt-driven systems
| Feature   | Polling                        | Interrupt-driven                    |
| --------- | ------------------------------ | ----------------------------------- |
| Mechanism | CPU checks status periodically | Hardware notifies CPU automatically |
| CPU usage | Wastes CPU cycles              | Efficient                           |
| Latency   | Depends on polling rate        | Very low                            |
| Use case  | Simple systems                 | Real-time systems                   |


**Q:** What is interrupt latency?
Interrupt latency is the delay between the interrupt event and the start of the ISR.
It includes:
- Time to complete current instruction
- Context save time
- Interrupt masking delays
- You want it as small and predictable as possible in real-time systems.


**Q:** What happens if interrupts nest?
- If nested interrupts are allowed (by priority), a higher-priority interrupt can preempt a running ISR.
- This improves responsiveness but increases complexity.
- Requires careful stack usage and predictable behavior.


**Q:** What restrictions apply to ISRs?
ISRs must:
- Be short and fast
- Avoid blocking functions (no printf(), malloc(), or delays)
- Not use APIs that can sleep or block
- Avoid shared data without synchronization


**Q:** Why should an ISR be short?
- Long ISRs block other interrupts, increase latency, and can break real-time guarantees.
- Best practice: do only the minimal work (like setting a flag), and let the main loop/task handle the rest.


**Q:** Why variables shared with ISR should be volatile?
volatile tells the compiler:
- “This variable can change unexpectedly (outside normal code flow). Don’t optimize it away.”
- Without volatile, the compiler might cache the variable in a register, and miss updates done by the ISR.


**Q:** Can you use a mutex or semaphore inside ISR?
- Mutex: ❌ No — it may block.
- Semaphore: ✅ Only if RTOS supports it via special API, e.g. xSemaphoreGiveFromISR() in FreeRTOS.
- Never use normal (task-level) APIs in ISR context.


**Q:** How to communicate between ISR and main code?
Common methods:
- Global flags or volatile variables
- Ring buffer
- Message queue (FromISR variant in RTOS)
- Event flags or signals

Example (bare metal):
```
volatile int data_ready = 0;

void ISR(void) {
    data_ready = 1;
}

int main(void) {
    while (1) {
        if (data_ready) {
            data_ready = 0;
            process_data();
        }
    }
}
```


**Q:** How are ISRs handled in an RTOS?
The RTOS distinguishes ISR context from task context.
ISRs:
- Run at a higher priority than tasks.
- Can’t use blocking RTOS APIs.
- Can trigger context switches (using yield or FromISR() APIs).


**Q:** What are FromISR() functions in FreeRTOS?
These are ISR-safe versions of standard APIs:
- xQueueSendFromISR()
- xSemaphoreGiveFromISR()
- xTaskNotifyFromISR()
They ensure no blocking occurs and manage context switching properly.


**Q:** How does context switching relate to ISRs?
- ISRs can trigger a higher-priority task to become ready.
- At the end of ISR, the RTOS checks —“Is there a higher-priority task ready now?”
- If yes → perform a context switch immediately before returning from ISR.


**Q:** Difference between task context and ISR context?
| Feature      | Task Context        | ISR Context                |
| ------------ | ------------------- | -------------------------- |
| Can block    | ✅ Yes              | ❌ No                      |
| Uses stack   | Task-specific stack | Shared ISR stack           |
| RTOS API use | Normal APIs         | Only `FromISR()` APIs      |
| Scheduling   | Controlled by RTOS  | Preempts tasks immediately |


**Q:** How does the CPU know which ISR to execute?
- Through the interrupt vector table —
- a table mapping interrupt sources to their ISR addresses.
- Each interrupt source (timer, UART, GPIO) has a vector entry.


**Q:** What happens if multiple interrupts occur simultaneously?
- The CPU uses priority levels:
- Higher-priority interrupts are serviced first.
- Others are pending until CPU finishes or nesting is allowed.


**Q:** What is interrupt priority and nesting?
- Priority: Defines which interrupt is more important.
- Nesting: Allows higher-priority interrupts to preempt lower-priority ones.
- Proper nesting improves responsiveness but increases stack usage.


**Q:** What is interrupt masking?
- Masking = temporarily disabling interrupts.
- Used to prevent re-entrancy or protect critical sections.
- E.g., cli() / sei() in AVR or __disable_irq() / __enable_irq() in ARM Cortex.


**Q:** How to measure ISR latency?
- Use a GPIO toggle technique:
- Toggle a pin at interrupt event and inside ISR.
- Measure delay with oscilloscope or logic analyzer.
- Latency = time between interrupt event and ISR start.


**Q:** What happens if ISR takes too long?
- Increases latency for other interrupts.
- Might cause missed deadlines.
- Can starve lower-priority tasks.
- May trigger watchdog reset.


**Q:** What if interrupt occurs inside another ISR?
Depends on nesting setting:
- If disabled → new interrupt stays pending.
- If enabled → CPU saves current context and jumps to higher-priority ISR.


**Q:** Common ISR pitfalls
- Forgetting to clear interrupt flag → repeated triggering.
- Using non-volatile globals → stale data.
- Using blocking code inside ISR → deadlock or crash.
- Long ISR → missed interrupts.


**Q:** How to debug ISR issues?
- Toggle GPIO at entry/exit to see ISR timing.
- Use debugger to view vector table or flags.
- Check interrupt enable bits and priorities.
- Verify flags are cleared correctly.