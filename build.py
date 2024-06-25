def build():

    print("Build csrc")
    print("Building with {}".format(sys.version_info))

    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    extensions_dir = this_dir/'nndet'/'csrc'

    main_file = list(extensions_dir.glob('*.cpp'))
    source_cpu = []  # list((extensions_dir/'cpu').glob('*.cpp')) temporary until I added header files ...
    source_cuda = list((extensions_dir/'cuda').glob('*.cu'))
    print("main_file {}".format(main_file))
    print("source_cpu {}".format(source_cpu))
    print("source_cuda {}".format(source_cuda))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []
    extra_compile_args = {"cxx": []}

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        print("Adding CUDA csrc to build")
        print("CUDA ARCH {}".format(os.getenv("TORCH_CUDA_ARCH_LIST")))
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        
        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [str(extensions_dir)]

    ext_modules = [
        extension(
            'nndet._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    distribution = Distribution({'name': 'extended', 'ext_modules': ext_modules})
    distribution.package_dir = 'extended'

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)

if __name__ == '__main__':
    build()