===============================================================================
CATEGORY 1: CHAT & GENERAL INTELLIGENCE (8 Prompts)
===============================================================================

1. Explain the concept of "monetary policy" and how central banks use interest rates to control inflation, using an analogy involving a thermostat.

2. Act as a neutral political analyst. Summarize the core arguments for and against universal basic income (UBI), highlighting the economic theories backing both sides.

3. Critique the following statement: "History is written by the victors." Provide historical examples that support this claim and at least two distinct examples that challenge it.

4. Write a professional, polite, yet firm email to a client explaining that their project scope has expanded significantly and that a budget increase of 25% is required before proceeding with the next phase.

5. Compare the philosophy of Stoicism with Epicureanism regarding their definitions of a "good life" and how one should handle personal adversity.

6. Provide a comprehensive summary of the major shifts in global supply chain logistics that occurred between 2020 and 2026, focusing on nearshoring trends.

7. Synthesize a brief overview of quantum computing for a non-technical corporate executive, explicitly explaining why it does not simply mean "faster classical computers."

8. A user is upset because their flight was canceled, and they are missing a family wedding. Draft an empathetic customer support response that balances policy constraints with human compassion.

===============================================================================
CATEGORY 2: REASONING & LOGIC (8 Prompts)
===============================================================================

9. A room contains 5 people. Each person shakes hands with every other person exactly once. How many handshakes occur in total? Walk through your logical steps clearly.

10. A farmer needs to cross a river with a wolf, a goat, and a cabbage. His boat can only hold himself and one of the three items at a time. If left alone, the wolf will eat the goat, and the goat will eat the cabbage. How can he get everything across safely?

11. Analyze the following scenario: If all Glips are Bloops, and no Bloops are Snarks, is it logically possible for some Snarks to be Glips? Provide a formal deductive proof or breakdown.

12. Five runners (A, B, C, D, E) finish a race. A finishes ahead of B but behind C. D finishes ahead of E but behind B. What is the exact finishing order of all five runners from first to last?

13. A specific virus population doubles every hour. If it takes exactly 48 hours for the population to completely fill a petri dish, how many hours does it take for the dish to be exactly half full? Explain the mathematical rationale.

14. You have two hourglasses: one measures exactly 4 minutes, and the other measures exactly 7 minutes. How can you use them to measure exactly 9 minutes sequentially?

15. If a clock shows the time as exactly 3:15, what is the precise angle in degrees between the hour hand and the minute hand? Show your calculation.

16. Evaluate the strategic vulnerability in this game theory setup: Two competing firms must choose whether to price their product high or low. If both choose high, both earn $5M. If both choose low, both earn $2M. If one chooses high and the other low, the low-pricing firm earns $8M and the high-pricing firm earns $0. Identify the Nash equilibrium.

===============================================================================
CATEGORY 3: CODING, ALGORITHMS, & OPTIMIZATION (24 Prompts)
===============================================================================

--- Python ---
17. Write a Python function that accepts a string containing nested parentheses, brackets, and braces `(), [], {}` and returns a boolean indicating whether the input string is syntactically valid/balanced.

18. Implement an asynchronous Python script using `asyncio` and `aiohttp` that concurrently fetches JSON data from five different mock endpoints, catches any HTTP errors gracefully, and aggregates the results into a single dictionary.

19. Review this Python function for a memory leak or inefficiency:
    def process_data(large_list):
        result = []
        for item in large_list:
            if item not in result:
                result.append(item)
        return result
    Explain why it performs poorly on large inputs and provide an optimized O(n) version.

20. Write a Python script using pandas to read a CSV file with columns 'Timestamp', 'UserID', and 'Action'. Find the top 3 users with the highest number of daily active interactions over a 30-day window.

--- Go ---
21. Write a concurrent worker pool framework in Go. The program should distribute a slice of 100 integers across 5 worker goroutines using channels, compute the factorial of each number, and collect the results safely.

22. Debug the following Go code block. Identify the race condition or issue, explain why it happens, and provide the corrected version:
    func main() {
        var wg sync.WaitGroup
        count := 0
        for i := 0; i < 1000; i++ {
            wg.Add(1)
            go func() {
                count++
                wg.Done()
            }()
        }
        wg.Wait()
        fmt.Println(count)
    }

23. Implement a custom HTTP middleware in Go that acts as a rate limiter using the token bucket algorithm, restricting IP addresses to a maximum of 10 requests per minute.

24. Write a Go function that takes two sorted slices of integers and merges them into a single sorted slice without using any built-in sorting libraries. Optimize for time and memory.

--- Swift ---
25. Write an elegant Swift implementation of a generic Stack data structure using an enum or a struct. Include `push`, `pop`, `peek`, and `isEmpty` capabilities, adhering to modern Swift style guidelines.

26. Convert this traditional closure-based completion handler code in Swift to use modern `async/await` structural concurrency patterns:
    func fetchUserProfile(id: String, completion: @escaping (Result<User, Error>) -> Void)

27. Write a Swift function using the Combine framework (or Actors) that observes changes to a text field, debounces the input by 300 milliseconds, filters out strings shorter than 3 characters, and prints the final query string.

28. Fix this Swift memory leak scenario: Explain the difference between `unowned` and `weak` references when resolving a strong reference cycle inside a closure capturing `self`.

--- Rust ---
29. Implement a thread-safe Binary Search Tree (BST) in Rust. Ensure it compiles cleanly under the strict ownership model, using appropriate smart pointers (`Rc`, `Arc`, `RefCell`, or `Mutex`) where necessary.

30. Write a Rust function that reads a file line by line, parses each line into an integer, handles missing or malformed data using the `Result` type without crashing (`panic!`), and returns the total sum.

31. Optimize this Rust function to minimize heap allocations and avoid unnecessary cloning of data:
    fn process_words(words: Vec<String>) -> Vec<String> {
        words.iter().map(|w| w.trim().to_uppercase()).collect()
    }
    Rewrite it to accept and return string slices where appropriate.

32. Explain how the Rust borrow checker handles the lifetimes in a function that takes two string references and returns a reference to the longest one. Provide the correctly annotated code signature.

--- Algorithms & Complex Data Structures ---
33. Implement Dijkstra's algorithm from scratch in the programming language of your choice. Provide clear type definitions for the graph nodes and edges.

34. Design an LRU (Least Recently Used) Cache data structure with `get(key)` and `put(key, value)` methods. Both operations must run in O(1) time complexity. Explain your architectural choice.

35. Given an array of integers representing daily stock prices, write an algorithm to find the maximum profit you could achieve by buying and selling that stock multiple times (you must sell before you buy again).

36. Write a function that solves the classic "N-Queens" puzzle using backtracking, returning all valid board configurations for an N x N grid.

--- Debugging & Optimizations ---
37. The following SQL query is executing slowly on a database with millions of rows:
    SELECT * FROM orders WHERE status = 'COMPLETED' AND order_date > '2025-01-01' ORDER BY total_amount DESC LIMIT 10;
    Suggest at least three concrete indexing strategies or query modifications to speed it up.

38. A web service handles massive JSON payloads. Profiling reveals that string allocations during JSON parsing are bottlenecking the CPU. Propose three structural architectural patterns to alleviate this.

39. Identify the logical edge cases missing from a function designed to parse URL strings into components (scheme, host, path, query parameters). List inputs that could break naive implementations.

40. Refactor a complex, deeply nested block of 5 conditional `if-else` loops into a clean, flat architecture using guard clauses, early returns, or pattern matching.

===============================================================================
CATEGORY 4: GERMAN LANGUAGE (10 Prompts)
===============================================================================

41. Übersetze den folgenden Text präzise ins Deutsche, wobei ein professioneller, geschäftsmäßiger Ton beibehalten werden muss: "We are thrilled to announce our expansion into the European market. Our team has worked tirelessly to ensure our software meets all regional compliance standards."

42. Verfasse eine formelle E-Mail auf Deutsch, in der du dich für eine Stelle als Senior Software Engineer bewirbst. Beziehe dich auf langjährige Erfahrung mit verteilten Systemen und Cloud-Architektur.

43. Erkläre den grammatikalischen Unterschied zwischen den deutschen Konjunktionen "weil", "denn" und "da". Gib für jede Konjunktion ein konkretes Beispiel und erläutere die jeweilige Wortstellung im Satz.

44. Korrigiere und optimiere den folgenden deutschen Text im Hinblick auf Stil, Rechtschreibung und Kommasetzung: "Der Kunde der gestern angerufen hatte wollte wissen, ob das Produkt lieferbar ist weil im Onlineshop steht das es ausverkauft sei."

45. Führe eine kurze literarische Analyse des Gedichts "Der Erlkönig" von Johann Wolfgang von Goethe durch. Was sind die zentralen Motive und wie wird die Spannung sprachlich aufgebaut?

46. Schreibe eine Zusammenfassung (ca. 200 Wörter) über die wirtschaftliche Bedeutung des deutschen "Mittelstands" im Vergleich zu Großkonzernen.

47. Simuliere ein kurzes Kundenservicedialog auf Deutsch: Ein Kunde beschwert sich höflich, aber bestimmt darüber, dass seine Rechnung fälschlicherweise doppelt abgebucht wurde. Reagiere als Support-Mitarbeiter lösungsorientiert.

48. Was bedeutet der juristische Begriff "Verhältnismäßigkeitsprinzip" im deutschen Recht? Erkläre die drei Stufen (Geeignetheit, Erforderlichkeit, Angemessenheit) anhand eines Beispiels.

49. Schreibe eine technische Dokumentation auf Deutsch für eine REST-API-Endpunkt, der Benutzerdaten aktualisiert (`PATCH /api/v1/users/:id`). Beschreibe Parameter, Validierungsregeln und HTTP-Statuscodes.

50. Diskutiere die Vor- und Nachteile der deutschen Energiewende aus ökonomischer und ökologischer Sicht. Bleibe dabei objektiv und sachlich.