{
  description = "Flake for fp64-emu kernel";

  inputs = {
    builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      builder,
    }:
    builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
