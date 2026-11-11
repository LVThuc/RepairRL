# RepairRL — bản thảo vòng cải thiện tiếp theo

Ngày: 2026-11-11. Trạng thái: **đề xuất, chưa triển khai hoặc chạy experiment mới**.
ETA tính từ lúc bắt đầu implement; giả định một RTX 3060 12 GB được dùng liên tục,
không có job GPU khác. Không cam kết tăng accuracy hoặc hoàn thành đúng cận dưới.

## Mục tiêu và các checkpoint

Kiểm tra ba thay đổi: controller có diagnosis và proposed change rõ ràng;
DPO học từ bản sửa sai thực tế của M1; GRPO chạy dài hơn với tín hiệu reward bớt
bị triệt tiêu. Giữ nguyên M0, M1 và kết quả cũ. Không train lại SFT trong vòng này.

```text
M0 (base) → M1 (SFT đã có)
             ├─ M2-hard: adapter DPO mới, hard negatives từ M1
             └─ M3-long: adapter GRPO mới, 600 updates
```

M2-hard và M3-long cùng bắt đầu từ M1-merged, không nối tiếp nhau.
Đánh giá mới dùng V3.1 cho cả bốn checkpoint; bảng V1 cũ được lưu riêng.

## 1. Agent V3.1: diagnosis → proposed change → repair → test

Ngân sách tối đa mỗi task: **3 lần sinh code và 2 lần diagnosis**. Diagnosis và
proposed change nằm trong cùng một JSON response, không phát sinh call thứ ba
ngoài hai diagnosis đã cấp. Attempt 1 giữ prompt repair ban đầu.

Sau mỗi candidate thất bại:

1. Ghi nhận số test pass, loại lỗi và exception/line đã được allowlist, redaction.
2. Diagnosis trả JSON `fault`, `evidence`, `change`; `change` phải chỉ rõ điều kiện,
   biểu thức, biến hoặc vòng lặp cần sửa và hành vi thay thế.
3. Đưa proposed change vào prompt repair như một giả thuyết cần thực hiện/kiểm tra.
   Không bỏ plan âm thầm khi nén context; rút history trước. Nếu vẫn không đủ
   context, ghi lỗi rõ ràng, không mở thêm call.
4. So sánh AST fingerprint với các candidate đã có. Patch lặp vẫn tiêu tốn một
   code-generation attempt, nhưng không được chạy lại để tạo thành tích giả.
5. Test candidate mới, lưu diff thực tế, kết quả test và mức tăng/giảm so với best.
6. Giữ best feedback-tested candidate. Diagnosis tiếp theo nhận cả plan trước,
   diff vừa thử và kết quả thất bại/regression, kể cả khi tiếp tục sửa từ best.

Diagnosis tối đa 128 tokens/call; code 512 tokens/call. Context đánh giá đề xuất
3,072 tokens cho tất cả checkpoint để có chỗ cho plan/history. Retry temperature
policy phải được chốt sau dev pilot, trước held-out evaluation, và áp dụng như
nhau cho cả bốn checkpoint. Không thay policy theo model hoặc kết quả test.

Plan là giả thuyết, không phải chain-of-thought được bảo đảm đúng. Log diff và
AST-change chỉ chứng minh code đã thay đổi; không chứng minh mô hình hiểu nguyên
nhân hoặc thực hiện đúng ngữ nghĩa của plan. Test mới quyết định repair success.

Dev pilot: 20–30 problem IDs cố định từ dev, chọn trước khi xem output; so sánh
V1 và V3.1 với cùng M1. Đo extra repairs sau attempt 1, duplicate rate, diagnosis
JSON validity, plan retention, regression và token/time overhead. Chỉ sửa controller
trên dev; sau đó đóng băng source/config cho đánh giá cuối. Không hứa V3.1 sẽ thắng.

## 2. M2-hard: DPO với candidate sai do M1 sinh

Dữ liệu gốc giữ nguyên: 2,000 train pairs/1,219 problems; 200 dev pairs;
200 held-out pairs. HumanEvalPack không dùng để sinh negatives hoặc chọn tham số.

Đề xuất chọn tối đa 600 problem IDs train bằng seed cố định, một source row/problem.
M1 sinh hai candidate/problem: một greedy và một stochastic. Lưu chính xác source
row ID, model/adapter hashes, seed, sampling settings, raw output, grade và test hash.
Được phép dùng published gold fix làm chosen; không cần sinh chosen mới.

Một hard-negative pair được giữ khi:

- Candidate parse được, là một chương trình đầy đủ; mọi test kết thúc bình thường
  nhưng có ít nhất một incorrect output. Timeout/runtime error không vào nhóm hard
  chính của vòng này.
- Candidate khác AST với original bug và chosen gold; bỏ bản lặp giữa các samples.
- Gold được xác nhận all-pass bằng cùng sandbox/suite. Không gọi đây là chứng minh
  đúng trên mọi input ngoài suite đã chọn.
- Chosen/rejected và prompt nằm trong context 2,048, không truncate mất phần sửa.

Nếu có hai negative hợp lệ, chọn candidate pass nhiều test hơn; tie-break cố định.
Target **300 pairs**; ngưỡng tối thiểu đề xuất **128 pairs**. Đây là target, không
phải số lượng đã có. Nếu hết 600 problems vẫn không đủ, ghi gate failed; không lấp
bằng original-bug weak negatives rồi gọi toàn bộ dữ liệu là hard negatives.

Training: fresh LoRA trên M1-merged; rank 16/alpha 32; sigmoid DPO beta 0.1;
learning rate 5e-6; 1 epoch; microbatch 1, accumulation 8; FP16/NF4; frozen reference
M1-merged, reference log-prob cache. Khoảng 38 updates nếu có 300 pairs. Lưu adapter
riêng `M2-hard`; không ghi đè M2 cũ. Chưa mặc định tăng epoch trên tập nhỏ.

## 3. M3-long: GRPO 600 steps và reward mới

Fresh branch từ M1-merged; không tiếp tục M2-hard và không dùng pilot làm main run.
Giữ max prompt 1,536 và max completion 512; 4 generations/group, microbatch 1,
accumulation 4; 600 optimizer steps, xấp xỉ 2,400 completions. Learning rate đề xuất
3e-6, beta KL 0.04; sampling temperature 0.8/top-p 0.8/top-k 20. Giữ QLoRA hiện có.

Reward đề xuất cho chương trình parse được:

```text
p = số test pass / tổng số test
c = số test kết thúc bình thường / tổng số test
r = 0.85*p + 0.10*c + 0.05*I(all_pass)
```

Code không parse được nhận 0. Normal exit có thể vẫn incorrect output. Runtime
error/timeout/output-limit không nhận điểm normal exit trên test đó, nhưng không
xóa điểm của những test khác đã pass. All-pass = 1.0, luôn cao hơn candidate fail.
Reward này là giả thuyết cần kiểm tra; phần thưởng normal exit có thể ưu tiên code
kết thúc nhưng sai, vì vậy phải log riêng all-pass và partial pass thay vì chỉ reward.

Chọn train curriculum từ các rollout M1 trên train: ưu tiên nhóm có reward khác
nhau, đồng thời giữ một phần task khác để giảm thu hẹp phân phối. Tỷ lệ đề xuất
70% mixed/30% task còn lại, tối đa 400 distinct problems; lưu IDs/hash. Nếu thiếu
mixed tasks, không giả vờ đạt tỷ lệ: báo số thực và dùng tập gốc hoặc dừng theo gate.
Không chọn curriculum bằng kết quả held-out/HumanEvalPack.

Go/no-go trước RL: 30 distinct train problems × 4 completions, cùng sampling và
reward mới; ít nhất 20% nhóm có variance > 0. Sau đó pilot 20 updates + save/reload.
Nếu gate fail, dừng M3-long và báo nguyên nhân; không tự đổi sang reward khác để
chạy cho xong. Log mixed-group fraction, KL, entropy, validity, all-pass, partial
fraction, clipping và latency. Main run mới bắt đầu nếu pilot không lỗi/NaN/OOM.

Đây là gói thay đổi gồm số steps, reward và curriculum. Nếu kết quả tăng, chỉ được
kết luận về gói này; muốn quy nguyên nhân cho từng thành phần cần thêm ablation.

## 4. Giao thức đánh giá được đóng băng

- RunBugRun: giữ nguyên 200 pairs/144 problem IDs và selected test suites.
- HumanEvalPack Python: đủ 164 tasks, bảng riêng, không dùng tuning.
- Bốn checkpoint: M0, M1, M2-hard, M3-long, cùng controller V3.1 đã đóng băng.
- Cùng prompt, tokenizer, context 3,072, code cap 512, diagnosis cap 128,
  maximum 3 code calls/2 diagnosis calls, seed policy, retry sampling, sandbox.
- Không cho model xem hidden test input, expected output hoặc test source.
- Công bố pass@1, agent@3, extra repairs, validity, duplicate rate, best-candidate
  fraction, generation/diagnosis tokens, wall time, attempts, tests run.
- Log mọi task, không chỉ demo thành công. Demo được tuyển chọn phải ghi rõ.
- Báo paired wins/losses và uncertainty; RunBugRun có nhiều pair cùng problem nên
  uncertainty phải cluster theo problem_id. Không coi mọi pair độc lập.

Nguồn/code/config/data/checkpoint hashes được chốt trước khi chạy held-out. Không
điều chỉnh rồi giữ bảng có kết quả đẹp nhất. Nếu có bug trong harness, đánh dấu
run bị vô hiệu và chạy lại mọi arm chịu ảnh hưởng bằng cùng giao thức sửa lỗi.
Giữ kết quả V1 cũ riêng; không so V3.1 model mới với V1 model cũ rồi quy hết cho RL.

## ETA trên RTX 3060 hiện tại

Các thời gian đã đo của vòng trước: SFT 26 phút; DPO 1,000 weak pairs 36.7 phút;
GRPO 200 updates 55.8 phút. V1 evaluation cả hai benchmark: M0 64.1 phút,
M1 99.5 phút, M2 98.0 phút, M3 95.7 phút: tổng gần 6 giờ. V3.1 sẽ có thêm diagnosis
và context dài hơn. Pilot V3 cũ chỉ 5 dev tasks, không đủ để suy ra ETA chính xác.

| Bước | ETA dự kiến | Đầu ra |
|---|---:|---|
| Implement V3.1, hard-pair builder, reward và supervisor; tests | 1.5–2.5 giờ | Code/CLI đã kiểm tra |
| V3 dev pilot + freeze protocol | 20–40 phút | Bảng V1/V3.1 trên dev, config cố định |
| M1 rollouts và verify hard negatives | 1–2.5 giờ | Target 300 pairs, audit/provenance |
| DPO hard + save/reload | 15–25 phút | M2-hard nếu data gate đạt |
| GRPO variance/pilot + 600-step main + reload | 3–4 giờ | M3-long nếu gates đạt |
| Full fixed evaluation: 4 models × 2 benchmarks, V3.1 | 8–10 giờ | Raw traces và đủ bảng kết quả |
| Tổng hợp báo cáo, paired comparison | 20–40 phút | Kết luận và giới hạn |
| **Tổng dự kiến** | **14–21 giờ** | Chưa tính lỗi cần sửa hoặc gate failure |

Mốc đầu tiên có thể xem được: code/controller dev check sau khoảng 2–3 giờ;
hard-DPO adapter sau khoảng 3–6 giờ; GRPO adapter sau khoảng 6–10 giờ; đầy đủ
aggregate evaluation sau khoảng 14–21 giờ. Các mốc là dự toán, không phải job đang chạy.

Nếu cần bản xem nhanh, có thể định trước một subset evaluation 50 RunBugRun +
50 HumanEvalPack tasks cho cả bốn model, đưa kết quả sớm hơn khoảng 4–6 giờ so với
full evaluation. Phải ghi là subset, không thay thế bảng full hoặc chọn subset sau
khi xem output. Bản đề xuất chính vẫn là full 200 + 164.

## Cách chạy tự động khi triển khai

Một supervisor dùng GPU tuần tự, log/status theo từng stage, khóa chống hai job
chiếm GPU cùng lúc, outputs tách biệt, kiểm tra upstream exit code và artifacts.
Các stage hoàn thành và hash hợp lệ được giữ lại khi resume. OOM được nhận diện,
thử lại có giới hạn với allocator đã dùng thành công; không âm thầm giảm context,
đổi reward hoặc đổi evaluation protocol. DPO gate fail không làm mất M1 parent;
GRPO có thể chạy độc lập nếu prerequisites của chính nó đạt. Run thất bại hoặc
checkpoint thiếu phải hiện rõ trong báo cáo cuối, không được bỏ qua.

Hiện tại: không khởi chạy training/evaluation mới. Bản thảo này chưa phải pipeline
đã implement; các adapter và số đo hiện có vẫn là vòng thử nghiệm cũ.
