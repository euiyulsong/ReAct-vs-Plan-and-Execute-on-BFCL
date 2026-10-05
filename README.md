# ReAct vs Plan-and-Execute Router Experiment

## Experiment Setup

BFCL tool-selection task에서 다음 네 가지 전략을 비교했다.

- **Always ReAct**: 매 step마다 필요한 tool 하나를 선택하고, 더 이상 필요 없으면 `-`로 종료
- **Always Plan**: 실행 전에 필요한 tool set을 한 번에 선택
- **Router**: 질문과 available tools를 보고 `0=ReAct`, `1=Plan-and-Execute`를 선택
- **Static Gold**: Single-tool → ReAct, Multi-tool → Plan을 사용하는 oracle-style routing

Dataset:

- **Single-tool:** 50 examples
- **Multi-tool:** 50 examples
- **Total:** 100 examples

Router를 제외한 ReAct/Plan prompt, parser, model, max step, evaluation 로직은 모두 동일하게 유지했다.

---

## Overall Results

| Method | Route Acc. | Tool EM | Precision | Recall | F1 | Count Match | LLM Calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Always Plan** | 0.50 | **0.96** | **0.9767** | **0.99** | **0.9813** | **0.97** | **1.00** |
| Always ReAct | 0.50 | 0.92 | **0.9767** | 0.97 | 0.9677 | 0.93 | 2.49 |
| **Router** | **0.97** | **0.96** | **0.9767** | **0.99** | **0.9813** | **0.97** | 1.00* |
| Static Gold | 1.00 | **0.96** | **0.9767** | **0.99** | **0.9813** | **0.97** | cached |

Router는 **97% routing accuracy**를 기록했고, tool-selection 성능에서는 Always Plan 및 Static Gold와 동일한 최고 성능을 기록했다.

```text
Tool Exact Match
Always Plan   0.96
Router        0.96
Static Gold   0.96
Always ReAct  0.92
```

즉 Router 자체는 Single/Multi-tool 구조를 거의 완벽하게 구분했지만, **최종 tool-selection 성능은 Always Plan을 넘지는 못했다.**

---

## Router Accuracy

Router confusion matrix:

```text
             Predicted
             ReAct   Plan
Gold ReAct     47      3
Gold Plan       0     50
```

따라서:

```text
Single-tool routing accuracy = 94%
Multi-tool routing accuracy  = 100%
Overall routing accuracy     = 97%
```

특히 **50개의 multi-tool query를 모두 Plan-and-Execute로 정확하게 routing**했다.

Single-tool에서는:

```text
47 / 50 → ReAct
 3 / 50 → Plan
```

으로 3개의 false positive만 발생했다.

---

## Multi-Tool Results

| Method | Tool EM | Precision | Recall | F1 | Count Match | LLM Calls |
|---|---:|---:|---:|---:|---:|---:|
| **Always Plan** | **0.98** | **0.9933** | **1.00** | **0.9960** | **0.98** | **1.00** |
| Always ReAct | 0.90 | **0.9933** | 0.96 | 0.9687 | 0.90 | 2.94 |
| **Router** | **0.98** | **0.9933** | **1.00** | **0.9960** | **0.98** | 1.00* |
| Static Gold | **0.98** | **0.9933** | **1.00** | **0.9960** | **0.98** | cached |

Multi-tool task에서는 결과가 매우 명확하다.

```text
Tool Exact Match
Plan    0.98
Router  0.98
ReAct   0.90
```

Router가 모든 multi-tool query를 Plan으로 보냈기 때문에 **Always Plan과 완전히 동일한 성능**을 얻었다.

ReAct는 순차적으로 tool을 하나씩 선택하면서 평균 **2.94 LLM calls**를 사용했지만, Plan은 한 번의 호출로 필요한 tool set을 모두 선택했다.

따라서 현재 BFCL multi-tool setting에서는:

> **여러 tool requirement를 original request만 보고 미리 식별할 수 있다면, upfront planning이 iterative ReAct보다 더 정확하고 효율적이다.**

---

## Single-Tool Results

| Method | Tool EM | Precision | Recall | F1 | Count Match | LLM Calls |
|---|---:|---:|---:|---:|---:|---:|
| Always Plan | **0.94** | **0.96** | **0.98** | **0.9667** | **0.96** | **1.00** |
| Always ReAct | **0.94** | **0.96** | **0.98** | **0.9667** | **0.96** | 2.04 |
| Router | **0.94** | **0.96** | **0.98** | **0.9667** | **0.96** | 1.00* |
| Static Gold | **0.94** | **0.96** | **0.98** | **0.9667** | **0.96** | cached |

Single-tool에서는 **Plan과 ReAct의 quality 차이가 전혀 없었다.**

```text
Tool EM
Plan    0.94
ReAct   0.94
Router  0.94

Tool F1
Plan    0.9667
ReAct   0.9667
Router  0.9667
```

하지만 LLM calls는:

```text
Plan   1.00
ReAct  2.04
```

로 Plan이 훨씬 적었다.

즉 이 benchmark에서는 single-tool조차 ReAct의 iterative loop가 추가적인 성능 향상을 만들지 못했다.

---

## Why Does Plan Work So Well?

이번 BFCL task의 중요한 특성은 **tool execution 결과를 실제로 관찰하지 않는다는 점**이다.

문제 구조는 사실상:

```text
User Request
     ↓
Available Tools
     ↓
Which tool(s) are required?
```

에 가깝다.

즉 필요한 tool들이 original request에 이미 드러나 있으므로:

```text
Plan
Question
 ↓
Required Tool Set
 ↓
Execute
```

가 잘 동작한다.

반면 ReAct의 핵심 장점은 다음처럼 **intermediate observation이 다음 action을 바꾸는 상황**에서 나타난다.

```text
Tool A
  ↓
Observation
  ↓
Tool B 결정
  ↓
Observation
  ↓
Tool C 결정
```

현재 BFCL experiment에는 이러한 observation dependency가 없다.

따라서 ReAct는 추가 LLM call을 사용하지만, 그 추가 step에서 새로운 정보가 생기지 않는다.

---

## Did the Router Help?

### Routing 관점

Router 자체는 매우 잘 동작했다.

```text
Route Accuracy = 97%
```

특히:

```text
Multi-tool → Plan
50 / 50 correct
```

였다.

따라서 질문과 tool descriptions만 보고 **single-tool vs multi-tool 구조를 판단하는 것은 GPT-6 Luna에게 상당히 쉬운 task**였다.

### End-to-End 관점

하지만 **Router를 추가할 실질적인 이점은 없었다.**

```text
Tool EM
Always Plan  0.96
Router       0.96
```

Router가 거의 완벽한 routing을 했음에도 Always Plan과 최종 quality가 동일했다.

그 이유는 single-tool에서도:

```text
Plan = ReAct = 0.94 EM
```

이었기 때문이다.

즉 Router가 정확하게 ReAct를 선택하더라도 얻는 추가적인 quality 이득이 없다.

---

## Important Finding: Routing Accuracy != System Improvement

이번 결과에서 가장 중요한 점이다.

Router는:

```text
97% routing accuracy
```

를 기록했다.

하지만:

```text
Router Tool EM = 0.96
Always Plan EM = 0.96
```

이었다.

따라서:

> **A highly accurate router does not necessarily improve the end-to-end system.**

Router가 가치 있으려면 두 execution strategy 사이에 실제로 **서로 다른 winning regions**가 있어야 한다.

예를 들어:

```text
Query A → ReAct wins
Query B → Plan wins
Query C → ReAct wins
...
```

같은 구조가 충분히 존재해야 Router가 best fixed strategy를 넘어설 수 있다.

현재 BFCL setting에서는 Plan이 거의 dominant strategy이므로 routing 자체의 가치가 낮다.

---

## Comparison with Retrieval Experiment

이전 retrieval experiment에서는 반대 결과가 나타났다.

### Multi-hop Retrieval

```text
Search
 ↓
Observation
 ↓
Follow-up Query
 ↓
Search
```

HotpotQA에서는 ReAct가 Plan-ReAct보다 높은 Full Support를 기록했다.

```text
ReAct       0.76
Plan-ReAct  0.67
```

### Tool Selection

이번 BFCL에서는:

```text
Multi-tool EM

Plan   0.98
ReAct  0.90
```

으로 Plan이 크게 우세했다.

두 결과를 함께 보면 execution strategy를 선택하는 더 일반적인 기준을 얻을 수 있다.

```text
Can future actions be determined from the original request?
    YES
    → Plan / direct tool-set selection

Does the next action depend on intermediate observations?
    YES
    → ReAct
```

즉 **single vs multi**, 또는 **single-hop vs multi-hop** 자체보다:

> **Does the next action depend on an intermediate observation?**

이 더 중요한 기준이다.

---

## Efficiency

관측된 평균 LLM calls:

```text
Always Plan   1.00
Always ReAct  2.49
Router        1.00 + cached execution
```

Plan은 전체 필요한 tool set을 한 번에 생성하므로 ReAct보다 훨씬 적은 LLM call을 사용했다.

특히 multi-tool:

```text
Plan   1.00 calls
ReAct  2.94 calls
```

로 차이가 크다.

따라서 현재 BFCL setting에서는 **quality와 inference efficiency 모두 Plan 쪽이 우세**하다.

---

## Latency Caveat

현재 실험에서는 cross-method LLM cache가 활성화되어 있다.

Router:

```text
llm_calls      = 1.00
llm_cache_hits = 1.47
latency        = 1.26s
```

Router가 실제 production에서 1.26초만 걸린다는 의미가 아니다.

Router 실행 후 선택된 ReAct/Plan 결과 상당 부분이 앞선 baseline 실행에서 cache되어 재사용되었다.

따라서 현재 결과에서는:

- **Tool quality 비교:** 유효
- **Router route accuracy:** 유효
- **Router standalone latency/cost:** 비교 불가

공정한 latency 측정을 위해서는:

```bash
python3 bfcl_router_fair_compare.py \
  --n-single 50 \
  --n-multi 50 \
  --max-react-steps 3 \
  --no-cache
```

로 별도 실행해야 한다.

실제 production에서는 Router가 항상 추가 호출이므로 대략:

```text
Router + Plan
≈ Router call + Plan call

Router + ReAct
≈ Router call + ReAct calls
```

의 비용이 발생한다.

---

## Conclusion

Router는 execution strategy를 매우 정확하게 분류했다.

```text
Route Accuracy = 97%

Single-tool:
47 / 50 → ReAct

Multi-tool:
50 / 50 → Plan
```

하지만 end-to-end 성능은:

```text
Tool Exact Match

Always Plan   0.96
Router        0.96
Always ReAct  0.92
```

으로 **Always Plan과 Router가 동일**했다.

따라서 현재 BFCL 환경에서는:

> **Router는 정확하지만 필요하지 않다.**

Router가 `single → ReAct`, `multi → Plan`을 거의 완벽하게 구분했음에도, single-tool에서 Plan과 ReAct의 quality가 동일했기 때문에 routing으로 추가적인 성능을 얻을 수 없었다.

현재 결과만 놓고 보면 가장 단순한 production policy는:

```text
BFCL-like tool selection
→ Always Plan / one-shot tool-set selection
```

이다.

반면 실제 tool output이 다음 action을 결정하는 환경에서는:

```text
Observation-dependent workflow
→ ReAct

Known tool requirements from initial request
→ Plan / direct tool-set selection
```

이라는 기준이 더 적절하다.

### Final Principle

> **Use planning when the required actions can be inferred upfront. Use ReAct when future actions depend on intermediate observations. Add a router only when different strategies actually win on meaningful subsets of production traffic.**
