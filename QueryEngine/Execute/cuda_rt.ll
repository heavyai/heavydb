target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() readnone nounwind
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() readnone nounwind
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() readnone nounwind

define i32 @pos_start_impl() {
  %threadIdx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %blockIdx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = mul nsw i32 %blockIdx, %blockDim
  %2 = add nsw i32 %threadIdx, %1
  ret i32 %2
}

define i32 @pos_step_impl() {
  %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %gridDim = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %1 = mul nsw i32 %blockDim, %gridDim
  ret i32 %1
}

