# ==============================================================================
# Este script resolve o problema de planejamento anual (12 meses) de forma
# sequencial mês a mês (Rolling Horizon / Fix-and-Optimize):
#   1. No mês t, recebe o estado herdado do mês t-1 (estoques nas fábricas,
#      POLs e PODs, cargas em trânsito marítimo despachadas em t-1 e t-2, e
#      navios ocupados de cada armador considerando o return time).
#   2. Resolve o subproblema MILP do mês t com uma janela curta de lookahead
#      (t+1, t+2) para orientar o envio de navios cujo transit time é de 1 ou 2
#      meses e a política de cobertura de 15 dias de estoque nos PODs.
#   3. Guarda as melhores decisões do mês t (produção, rotas/navios, estoques e
#      atendimento aos clientes), atualiza o estado e avança para o mês t+1.
# ==============================================================================

using Pkg

function garantir_pacotes(pacotes::Vector{String})
    instalados = keys(Pkg.project().dependencies)
    faltantes = [p for p in pacotes if !(p in instalados)]
    if !isempty(faltantes)
        println("Instalando pacotes necessários: ", join(faltantes, ", "))
        Pkg.add(faltantes)
    end
end

garantir_pacotes(["CSV", "DataFrames", "JuMP", "HiGHS"])

using CSV
using DataFrames
using JuMP
using HiGHS
using Printf
using Statistics

# Helper simples para OrderedSet sem pacote extra
struct OrderedSet{T}
    itens::Vector{T}
    conjunto::Set{T}
    OrderedSet{T}() where T = new{T}(T[], Set{T}())
end
function Base.push!(os::OrderedSet{T}, x) where T
    val = convert(T, x)
    if !(val in os.conjunto)
        push!(os.conjunto, val)
        push!(os.itens, val)
    end
    return os
end
Base.collect(os::OrderedSet) = copy(os.itens)

# ------------------------------------------------------------------------------
# 1. Parâmetros e Configurações do Modelo
# ------------------------------------------------------------------------------
Base.@kwdef mutable struct ConfigModelo
    diretorio_dados::String = @__DIR__
    diretorio_saida::String = joinpath(@__DIR__, "resultados")
    horizonte_planejamento::Int = 12       # Meses a resolver e fixar (t = 1..12)
    janela_lookahead::Int = 2              # Meses futuros visíveis no subproblema (max TEMPO_IDA = 2)
    dias_cobertura_pod::Float64 = 15.0     # Política de cobertura: 15 dias (0.5 mês)
    penalidade_cobertura::Float64 = 15.0   # Penalidade (USD/ton) por déficit de cobertura no POD
    penalidade_falta_demanda::Float64 = 0.0 # Se > 0, adiciona penalidade extra além da perda da receita
    custo_estocagem_mensal::Float64 = 0.05 # Pequeno custo de carregamento de estoque para evitar acúmulo ocioso
    usar_desconto_escalonado::Bool = true  # Desafio 2.4.2: Desconto por faixa (0-50%, 50-75%, 75-100%)
    usar_return_time::Bool = true          # Desafio 2.4.1: Considerar TEMPO_IDA_VOLTA na frota
    usar_limite_frota::Bool = true         # Ativar limite de frota por armador
    max_rotas_por_grupo::Int = 3           # Filtro de rotas dominadas por (Origens, POD, Posição, TempoIda)
    time_limit_por_mes::Float64 = 30.0     # Limite de tempo do solver HiGHS por mês (segundos)
    mip_rel_gap::Float64 = 0.005           # Gap relativo de otimalidade (0.5%) por subproblema mensal
    frota_max_armador::Dict{String,Int} = Dict(
        "AHAB"    => 30,
        "AMERICO" => 30,
        "CASPIAN" => 30,
        "COLOMBO" => 45,
        "NEMO"    => 25,
        "SINBAD"  => 35,
        "ULISSES" => 40
    )
end

# ------------------------------------------------------------------------------
# 2. Estruturas de Dados do Problema
# ------------------------------------------------------------------------------
struct RotaMaritima
    id::Int
    armador::String
    origens::Vector{String}
    destinos::Vector{String}
    min_intake::Float64
    max_intake::Float64
    frete_base::Float64
    tempo_ida::Int
    tempo_ida_volta::Int
end

struct DadosSuzano
    fabricas::Vector{String}
    pols::Vector{String}
    pods::Vector{String}
    armadores::Vector{String}
    clientes::Vector{String}
    produtos::Vector{Tuple{String,String}} # (FAMILIA_PRODUTO, LINHA_PRODUTO)
    prod_para_id::Dict{Tuple{String,String},Int}
    linha_do_prod::Vector{String}
    pol_da_fabrica::Dict{String,String}
    fabrica_da_linha::Dict{String,String}
    pol_da_linha::Dict{String,String}
    max_carregamento::Dict{String,Float64}
    max_descarregamento::Dict{String,Float64}
    max_estoque::Dict{String,Float64}
    calado_dwt::Dict{String,Float64}
    cap_fabrica::Dict{Tuple{String,Int},Float64}
    custo_inland::Dict{Tuple{String,String},Float64}
    pods_do_cliente::Dict{String,Vector{String}}
    demanda::Dict{Tuple{String,Int,Int},Float64}     # (cliente, prod_id, mes) -> ton
    preco_venda::Dict{Tuple{String,Int,Int},Float64} # (cliente, prod_id, mes) -> USD/ton
    est_inicial::Dict{Tuple{String,Int},Float64}     # (local, prod_id) -> ton
    rotas::Vector{RotaMaritima}
    max_instante_dados::Int
end

# ------------------------------------------------------------------------------
# 3. Funções Auxiliares de Leitura e Pré-processamento
# ------------------------------------------------------------------------------
function limpar_str(val)::String
    if ismissing(val) || val === nothing
        return ""
    else
        return String(strip(string(val)))
    end
end

function parse_num(val)::Float64
    if ismissing(val) || val === nothing
        return 0.0
    elseif val isa Real
        return Float64(val)
    else
        s = limpar_str(val)
        isempty(s) && return 0.0
        return parse(Float64, replace(s, "," => "."))
    end
end

function parse_bool(val)::Bool
    if ismissing(val) || val === nothing
        return false
    elseif val isa Bool
        return val
    else
        s = uppercase(limpar_str(val))
        return s in ("TRUE", "1", "SIM", "T", "YES")
    end
end

function carregar_dados(config::ConfigModelo)::DadosSuzano
    dir = config.diretorio_dados
    println("Carregando arquivos CSV de: ", dir)

    df_clientes  = CSV.read(joinpath(dir, "MDClientes.csv"), DataFrame; stringtype=String)
    df_locais    = CSV.read(joinpath(dir, "MDLocais.csv"), DataFrame; stringtype=String)
    df_produtos  = CSV.read(joinpath(dir, "MDProdutos.csv"), DataFrame; stringtype=String)
    df_cap_fab   = CSV.read(joinpath(dir, "RDCapFabricas.csv"), DataFrame; stringtype=String)
    df_inland    = CSV.read(joinpath(dir, "RDCustoInland.csv"), DataFrame; stringtype=String)
    df_demanda   = CSV.read(joinpath(dir, "RDDemanda.csv"), DataFrame; stringtype=String)
    df_est_ini   = CSV.read(joinpath(dir, "RDEstoqueInicial.csv"), DataFrame; stringtype=String)
    df_loc_prod  = CSV.read(joinpath(dir, "RDLocalProduto.csv"), DataFrame; stringtype=String)
    df_rotas     = CSV.read(joinpath(dir, "MDArmadorRotas.csv"), DataFrame; stringtype=String)

    # 1. Locais e capacidades operacionais
    fabricas = String[]
    pols = String[]
    pods = String[]
    max_carregamento = Dict{String,Float64}()
    max_descarregamento = Dict{String,Float64}()
    max_estoque = Dict{String,Float64}()
    calado_dwt = Dict{String,Float64}()

    for row in eachrow(df_locais)
        loc = limpar_str(row.COD_LOCAL)
        tipo = limpar_str(row.TIPO)
        max_carregamento[loc] = parse_num(row.MAX_CARREGAMENTO)
        max_descarregamento[loc] = parse_num(row.MAX_DESCARREGAMENTO)
        max_estoque[loc] = parse_num(row.MAX_ESTOQUE)
        calado_dwt[loc] = parse_num(row.CALADO_DWT)
        if tipo == "Fabrica"
            push!(fabricas, loc)
        elseif tipo == "POL"
            push!(pols, loc)
        elseif tipo == "POD"
            push!(pods, loc)
        end
    end

    # Relação fixa Fábrica -> POL (Seção 2.2 do enunciado)
    pol_da_fabrica = Dict{String,String}(
        "Suzano"     => "Santos",
        "TresLagoas" => "Santos",
        "Ribas"      => "Santos",
        "Limeira"    => "Santos",
        "Jacarei"    => "Santos",
        "Mucuri"     => "Portocel",
        "Aracruz"    => "Portocel",
        "Veracel"    => "Portocel",
        "Imperatriz" => "Itaqui"
    )

    # 2. Linhas de produto habilitadas por fábrica
    fabrica_da_linha = Dict{String,String}()
    pol_da_linha = Dict{String,String}()
    for row in eachrow(df_loc_prod)
        if parse_bool(row.PRODUZ)
            fab = limpar_str(row.COD_LOCAL)
            lin = limpar_str(row.LINHA_PRODUTO)
            fabrica_da_linha[lin] = fab
            pol_da_linha[lin] = pol_da_fabrica[fab]
        end
    end

    # 3. Produtos ativos (incluindo produtos presentes em demanda ou estoque inicial)
    produtos_set = OrderedSet{Tuple{String,String}}()
    for row in eachrow(df_produtos)
        if parse_bool(row.ATIVO)
            push!(produtos_set, (limpar_str(row.FAMILIA_PRODUTO), limpar_str(row.LINHA_PRODUTO)))
        end
    end
    for row in eachrow(df_demanda)
        push!(produtos_set, (limpar_str(row.FAMILIA_PRODUTO), limpar_str(row.LINHA_PRODUTO)))
    end
    for row in eachrow(df_est_ini)
        push!(produtos_set, (limpar_str(row.FAMILIA_PRODUTO), limpar_str(row.LINHA_PRODUTO)))
    end

    produtos = collect(produtos_set)
    prod_para_id = Dict{Tuple{String,String},Int}(p => i for (i, p) in enumerate(produtos))
    linha_do_prod = [p[2] for p in produtos]

    # 4. Clientes ativos
    clientes_ativos = Set{String}()
    for row in eachrow(df_clientes)
        if parse_bool(row.ATIVO)
            push!(clientes_ativos, limpar_str(row.COD_CLIENTE))
        end
    end
    clientes = sort(collect(clientes_ativos))

    # 5. Custos in-land (POD -> Cliente)
    custo_inland = Dict{Tuple{String,String},Float64}()
    pods_do_cliente = Dict{String,Vector{String}}(c => String[] for c in clientes)
    pods_set = Set(pods)
    for row in eachrow(df_inland)
        cli = limpar_str(row.COD_CLIENTE)
        pod = limpar_str(row.COD_LOCAL)
        if (cli in clientes_ativos) && (pod in pods_set)
            c_in = parse_num(row.CUSTO_INLAND_POR_TON)
            if !haskey(custo_inland, (cli, pod)) || c_in < custo_inland[(cli, pod)]
                custo_inland[(cli, pod)] = c_in
                if !(pod in pods_do_cliente[cli])
                    push!(pods_do_cliente[cli], pod)
                end
            end
        end
    end

    # 6. Capacidade líquida das fábricas por mês
    cap_fabrica = Dict{Tuple{String,Int},Float64}()
    max_instante = 12
    for row in eachrow(df_cap_fab)
        fab = limpar_str(row.COD_LOCAL)
        t = Int(round(parse_num(row.INSTANTE)))
        vol = parse_num(row.VOLUME_PRODUCAO)
        cap_fabrica[(fab, t)] = vol
        max_instante = max(max_instante, t)
    end

    # 7. Demanda e Preço de Venda
    demanda = Dict{Tuple{String,Int,Int},Float64}()
    preco_venda = Dict{Tuple{String,Int,Int},Float64}()
    for row in eachrow(df_demanda)
        cli = limpar_str(row.COD_CLIENTE)
        cli in clientes_ativos || continue
        fam = limpar_str(row.FAMILIA_PRODUTO)
        lin = limpar_str(row.LINHA_PRODUTO)
        p_id = prod_para_id[(fam, lin)]
        t = Int(round(parse_num(row.INSTANTE)))
        preco = parse_num(row.PRECO_VENDA)
        dem = parse_num(row.DEMANDA)
        if dem > 0
            chave = (cli, p_id, t)
            demanda[chave] = get(demanda, chave, 0.0) + dem
            preco_venda[chave] = preco
            max_instante = max(max_instante, t)
        end
    end

    # 8. Estoque Inicial (Fábricas, POLs e PODs)
    est_inicial = Dict{Tuple{String,Int},Float64}()
    for row in eachrow(df_est_ini)
        loc = limpar_str(row.COD_LOCAL)
        fam = limpar_str(row.FAMILIA_PRODUTO)
        lin = limpar_str(row.LINHA_PRODUTO)
        qtd = parse_num(row.QTD_DISPONIVEL)
        if qtd > 0
            p_id = prod_para_id[(fam, lin)]
            est_inicial[(loc, p_id)] = get(est_inicial, (loc, p_id), 0.0) + qtd
        end
    end

    # 9. Rotas Marítimas e Pré-seleção por Viabilidade de Calado (DWT) e Dominância
    todas_rotas = RotaMaritima[]
    armadores_set = Set{String}()
    for (idx, row) in enumerate(eachrow(df_rotas))
        arm = limpar_str(row.COD_ARMADOR)
        push!(armadores_set, arm)
        origs = unique([String(strip(s)) for s in split(limpar_str(row.COD_LOCAL_ORIGENS), ";") if !isempty(strip(s))])
        dests = unique([String(strip(s)) for s in split(limpar_str(row.COD_LOCAL_DESTINOS), ";") if !isempty(strip(s))])
        min_in = parse_num(row.MIN_INTAKE)
        max_in = parse_num(row.MAX_INTAKE)
        frete  = parse_num(row.FRETE_POR_TON)
        t_ida  = Int(round(parse_num(row.TEMPO_IDA)))
        t_idav = Int(round(parse_num(row.TEMPO_IDA_VOLTA)))

        # Verificação física de calado (DWT):
        # - O último POL da rota precisa comportar ao menos o MIN_INTAKE de saída do Brasil
        # - O primeiro POD da rota precisa comportar ao menos o MIN_INTAKE de chegada no exterior
        dwt_ult_orig = get(calado_dwt, origs[end], 0.0)
        dwt_prim_dest = get(calado_dwt, dests[1], 0.0)
        if dwt_ult_orig >= min_in && dwt_prim_dest >= min_in
            push!(todas_rotas, RotaMaritima(idx, arm, origs, dests, min_in, max_in, frete, t_ida, t_idav))
        end
    end

    # Seleção das melhores rotas (mantendo todas as rotas de destino único + top K por grupo composto)
    rotas_selecionadas_ids = Set{Int}()
    grupos = Dict{Tuple{String,String,Bool,Int},Vector{RotaMaritima}}()
    for r in todas_rotas
        if length(r.destinos) == 1
            push!(rotas_selecionadas_ids, r.id)
        end
        orig_str = join(r.origens, ";")
        for (pos, d) in enumerate(r.destinos)
            if get(calado_dwt, d, 0.0) > 0 && get(max_descarregamento, d, 0.0) > 0
                chave_g = (orig_str, d, pos == 1, r.tempo_ida)
                lista = get!(grupos, chave_g, RotaMaritima[])
                push!(lista, r)
            end
        end
    end
    for (_, lista) in grupos
        sort!(lista, by = r -> r.frete_base)
        for r in Iterators.take(lista, config.max_rotas_por_grupo)
            push!(rotas_selecionadas_ids, r.id)
        end
    end

    rotas_finais = [r for r in todas_rotas if r.id in rotas_selecionadas_ids]
    armadores = sort(collect(armadores_set))

    println(@sprintf("Dados carregados: %d fábricas, %d POLs, %d PODs, %d produtos, %d clientes ativos.",
        length(fabricas), length(pols), length(pods), length(produtos), length(clientes)))
    println(@sprintf("Rotas totais: %d | Viáveis no calado: %d | Selecionadas para otimização: %d.",
        nrow(df_rotas), length(todas_rotas), length(rotas_finais)))

    return DadosSuzano(
        fabricas, pols, pods, armadores, clientes, produtos, prod_para_id,
        linha_do_prod, pol_da_fabrica, fabrica_da_linha, pol_da_linha,
        max_carregamento, max_descarregamento, max_estoque, calado_dwt,
        cap_fabrica, custo_inland, pods_do_cliente, demanda, preco_venda,
        est_inicial, rotas_finais, max_instante
    )
end

# ------------------------------------------------------------------------------
# 4. Estado Dinâmico Entre os Meses e Estrutura de Decisões Mensais
# ------------------------------------------------------------------------------
mutable struct EstadoSistema
    estoque_fab::Dict{Tuple{String,Int},Float64}        # (fabrica, prod_id) -> ton ao final de t-1
    estoque_pol::Dict{Tuple{String,Int},Float64}        # (pol, prod_id) -> ton ao final de t-1
    estoque_pod::Dict{Tuple{String,Int},Float64}        # (pod, prod_id) -> ton ao final de t-1
    transito_pod::Dict{Tuple{String,Int,Int},Float64}   # (pod, prod_id, mes_chegada) -> ton já em trânsito no mar
    navios_ocupados::Dict{Tuple{String,Int},Int}        # (armador, mes) -> qtd navios ocupados vindos de meses < t
end

function inicializar_estado(dados::DadosSuzano)::EstadoSistema
    est_fab = Dict{Tuple{String,Int},Float64}()
    est_pol = Dict{Tuple{String,Int},Float64}()
    est_pod = Dict{Tuple{String,Int},Float64}()

    set_fab = Set(dados.fabricas)
    set_pol = Set(dados.pols)
    set_pod = Set(dados.pods)

    for ((loc, p_id), qtd) in dados.est_inicial
        if loc in set_fab
            est_fab[(loc, p_id)] = qtd
        elseif loc in set_pol
            est_pol[(loc, p_id)] = qtd
        elseif loc in set_pod
            est_pod[(loc, p_id)] = qtd
        end
    end

    return EstadoSistema(
        est_fab,
        est_pol,
        est_pod,
        Dict{Tuple{String,Int,Int},Float64}(),
        Dict{Tuple{String,Int},Int}()
    )
end

struct DecisoesMes
    mes::Int
    receita_bruta::Float64
    custo_frete::Float64
    custo_inland::Float64
    margem_operacional::Float64
    demanda_total::Float64
    demanda_atendida::Float64
    deficit_cobertura_total::Float64
    total_navios::Int
    producao::Vector{NamedTuple}
    embarques::Vector{NamedTuple}
    estoques_pol::Vector{NamedTuple}
    estoques_pod::Vector{NamedTuple}
    atendimento::Vector{NamedTuple}
end

# ------------------------------------------------------------------------------
# 5. Subproblema de Otimização do Mês `t_atual` (com Lookahead t_atual .. t_fim)
# ------------------------------------------------------------------------------
function resolver_mes!(
    t_atual::Int,
    dados::DadosSuzano,
    config::ConfigModelo,
    estado::EstadoSistema
)::DecisoesMes
    t_fim = min(dados.max_instante_dados, t_atual + config.janela_lookahead)
    janela = collect(t_atual:t_fim)
    n_prods = length(dados.produtos)

    # Identificar quais pares (POD, prod_id) são relevantes na rede
    pods_ativos_por_prod = Dict{Int,Set{String}}(p => Set{String}() for p in 1:n_prods)
    for ((cli, p, _), _) in dados.demanda
        for d in dados.pods_do_cliente[cli]
            push!(pods_ativos_por_prod[p], d)
        end
    end
    for ((d, p), qtd) in estado.estoque_pod
        qtd > 1e-4 && push!(pods_ativos_por_prod[p], d)
    end
    for ((d, p, _), qtd) in estado.transito_pod
        qtd > 1e-4 && push!(pods_ativos_por_prod[p], d)
    end

    # Qual(is) POL(s) podem expedir cada produto p?
    pols_do_prod = Dict{Int,Vector{String}}()
    for p in 1:n_prods
        lin = dados.linha_do_prod[p]
        pol_padrao = get(dados.pol_da_linha, lin, "Santos")
        lista_l = [pol_padrao]
        for l in dados.pols
            if l != pol_padrao && get(estado.estoque_pol, (l, p), 0.0) > 1e-4
                push!(lista_l, l)
            end
        end
        pols_do_prod[p] = unique(lista_l)
    end

    # Construir o modelo JuMP para a janela [t_atual .. t_fim]
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "time_limit", config.time_limit_por_mes)
    set_attribute(model, "mip_rel_gap", config.mip_rel_gap)

    # --- 5.1 Variáveis de Produção e Estoque nas Fábricas ---
    prods_da_fab = Dict{String,Vector{Int}}(f => Int[] for f in dados.fabricas)
    for p in 1:n_prods
        lin = dados.linha_do_prod[p]
        if haskey(dados.fabrica_da_linha, lin)
            push!(prods_da_fab[dados.fabrica_da_linha[lin]], p)
        end
        for f in dados.fabricas
            if get(estado.estoque_fab, (f, p), 0.0) > 1e-4 && !(p in prods_da_fab[f])
                push!(prods_da_fab[f], p)
            end
        end
    end
    pares_fp = Tuple{String,Int}[]
    for f in dados.fabricas
        unique!(prods_da_fab[f])
        for p in prods_da_fab[f]
            push!(pares_fp, (f, p))
        end
    end

    @variable(model, x_prod[fp in pares_fp, t in janela] >= 0)
    @variable(model, e_fab[fp in pares_fp, t in janela] >= 0)
    @variable(model, I_fab[fp in pares_fp, t in janela] >= 0)
    @variable(model, folga_carreg_fab[f in dados.fabricas, t in janela] >= 0)
    @variable(model, folga_est_fab[f in dados.fabricas, t in janela] >= 0)

    for f in dados.fabricas, t in janela
        cap_ft = get(dados.cap_fabrica, (f, t), 0.0)
        # 100% da capacidade líquida deve ser utilizada (apenas nos produtos que a fábrica produz)
        prods_hab = [p for p in prods_da_fab[f] if get(dados.fabrica_da_linha, dados.linha_do_prod[p], "") == f]
        @constraint(model, sum(x_prod[(f, p), t] for p in prods_hab) == cap_ft)
        for p in prods_da_fab[f]
            if !(p in prods_hab)
                @constraint(model, x_prod[(f, p), t] == 0.0)
            end
            est_ant = (t == t_atual) ? get(estado.estoque_fab, (f, p), 0.0) : I_fab[(f, p), t - 1]
            @constraint(model, I_fab[(f, p), t] == est_ant + x_prod[(f, p), t] - e_fab[(f, p), t])
        end
        @constraint(model, sum(e_fab[(f, p), t] for p in prods_da_fab[f]) <= dados.max_carregamento[f] + folga_carreg_fab[f, t])
        @constraint(model, sum(I_fab[(f, p), t] for p in prods_da_fab[f]) <= dados.max_estoque[f] + folga_est_fab[f, t])
    end

    # --- 5.2 Variáveis de Rotas Marítimas e Navios ---
    # No mês corrente (t == t_atual), n_navios é INTEIRO; nos meses de lookahead (t > t_atual),
    # relaxamos para contínuo para acelerar drasticamente a resolução mantendo o sinal exato de custo/capacidade.
    R_idx = 1:length(dados.rotas)
    @variable(model, n_navios[r in R_idx, t in janela] >= 0)
    for r in R_idx
        set_integer(n_navios[r, t_atual])
    end

    # Fluxo total (ton) na rota r do POL de origem l para o POD de destino d no mês t
    pares_rld = Tuple{Int,String,String}[]
    for (r_i, r) in enumerate(dados.rotas)
        for l in r.origens, d in r.destinos
            push!(pares_rld, (r_i, l, d))
        end
    end
    unique!(pares_rld)
    @variable(model, Y_rota[rld in pares_rld, t in janela] >= 0)

    # Faixas de desconto escalonado (0-50%, 50-75%, 75-100% de MAX_INTAKE)
    @variable(model, w_faixa[r in R_idx, faixa in 1:3, t in janela] >= 0)

    for (r_i, r) in enumerate(dados.rotas), t in janela
        carga_total_rt = sum(Y_rota[(r_i, l, d), t] for l in r.origens, d in r.destinos)

        # (a) Frete morto inadmissível: intervalo [MIN_INTAKE, MAX_INTAKE] por navio
        @constraint(model, carga_total_rt >= r.min_intake * n_navios[r_i, t])
        @constraint(model, carga_total_rt <= r.max_intake * n_navios[r_i, t])

        # (b) Faixas de volume para desconto escalonado
        @constraint(model, carga_total_rt == w_faixa[r_i, 1, t] + w_faixa[r_i, 2, t] + w_faixa[r_i, 3, t])
        @constraint(model, w_faixa[r_i, 1, t] <= 0.50 * r.max_intake * n_navios[r_i, t])
        @constraint(model, w_faixa[r_i, 2, t] <= 0.25 * r.max_intake * n_navios[r_i, t])
        @constraint(model, w_faixa[r_i, 3, t] <= 0.25 * r.max_intake * n_navios[r_i, t])

        # (c) Calado sequencial nos POLs de origem (peso acumulado a bordo ao sair de cada POL)
        for idx_l in 1:length(r.origens)
            pols_ate_aqui = r.origens[1:idx_l]
            dwt_l = dados.calado_dwt[r.origens[idx_l]]
            @constraint(model, sum(Y_rota[(r_i, l, d), t] for l in pols_ate_aqui, d in r.destinos) <= dwt_l * n_navios[r_i, t])
        end

        # (d) Calado sequencial nos PODs de destino (peso a bordo ao chegar em cada POD antes de descarregar)
        for idx_d in 1:length(r.destinos)
            pods_restantes = r.destinos[idx_d:end]
            dwt_d = dados.calado_dwt[r.destinos[idx_d]]
            @constraint(model, sum(Y_rota[(r_i, l, d), t] for l in r.origens, d in pods_restantes) <= dwt_d * n_navios[r_i, t])
        end
    end

    # Limite de frota por armador (com ou sem return time)
    @variable(model, folga_frota[a in dados.armadores, t in janela] >= 0)
    if config.usar_limite_frota
        rotas_do_armador = Dict{String,Vector{Int}}(a => Int[] for a in dados.armadores)
        for (r_i, r) in enumerate(dados.rotas)
            push!(rotas_do_armador[r.armador], r_i)
        end
        for a in dados.armadores, t in janela
            frota_max = get(config.frota_max_armador, a, 100)
            ocupados_hist = config.usar_return_time ? get(estado.navios_ocupados, (a, t), 0) : 0
            navios_janela = AffExpr(0.0)
            for r_i in rotas_do_armador[a]
                rho = config.usar_return_time ? dados.rotas[r_i].tempo_ida_volta : 1
                for t_part in janela
                    if t_part <= t && t < t_part + rho
                        add_to_expression!(navios_janela, 1.0, n_navios[r_i, t_part])
                    end
                end
            end
            @constraint(model, ocupados_hist + navios_janela <= frota_max + folga_frota[a, t])
        end
    end

    # --- 5.3 Fluxo Marítimo por Produto (l, d, tau, p, t) e Acoplamento com Rotas ---
    rotas_por_ldt = Dict{Tuple{String,String,Int},Vector{Int}}()
    for (r_i, r) in enumerate(dados.rotas)
        for l in r.origens, d in r.destinos
            chave = (l, d, r.tempo_ida)
            lista = get!(rotas_por_ldt, chave, Int[])
            push!(lista, r_i)
        end
    end

    tuplas_f = Tuple{String,String,Int,Int}[] # (l, d, tau, p)
    tuplas_f_por_ldt = Dict{Tuple{String,String,Int},Vector{Int}}()
    tuplas_f_por_lp  = Dict{Tuple{String,Int},Vector{Tuple{String,Int}}}()
    tuplas_f_por_dp_tau = Dict{Tuple{String,Int,Int},Vector{String}}()

    for p in 1:n_prods
        for l in pols_do_prod[p], d in pods_ativos_por_prod[p], tau in (1, 2)
            if haskey(rotas_por_ldt, (l, d, tau))
                push!(tuplas_f, (l, d, tau, p))
                push!(get!(tuplas_f_por_ldt, (l, d, tau), Int[]), p)
                push!(get!(tuplas_f_por_lp, (l, p), Tuple{String,Int}[]), (d, tau))
                push!(get!(tuplas_f_por_dp_tau, (d, p, tau), String[]), l)
            end
        end
    end
    unique!(tuplas_f)

    @variable(model, f_mar[ldtp in tuplas_f, t in janela] >= 0)

    # Acoplamento: a soma dos produtos embarcados em (l, d, tau) é igual à capacidade alocada nas rotas (l, d, tau)
    for ((l, d, tau), r_indices) in rotas_por_ldt, t in janela
        prods_ldt = get(tuplas_f_por_ldt, (l, d, tau), Int[])
        if isempty(prods_ldt)
            for r_i in r_indices
                @constraint(model, Y_rota[(r_i, l, d), t] == 0.0)
            end
        else
            @constraint(model,
                sum(f_mar[(l, d, tau, p), t] for p in prods_ldt) ==
                sum(Y_rota[(r_i, l, d), t] for r_i in r_indices)
            )
        end
    end

    # --- 5.4 Balanço de Estoque e Capacidades nos POLs ---
    @variable(model, I_pol[l in dados.pols, p in 1:n_prods, t in janela] >= 0)
    @variable(model, folga_est_pol[l in dados.pols, t in janela] >= 0)
    @variable(model, folga_desc_pol[l in dados.pols, t in janela] >= 0)

    fabs_do_pol = Dict{String,Vector{String}}(l => String[] for l in dados.pols)
    for (f, l) in dados.pol_da_fabrica
        push!(fabs_do_pol[l], f)
    end

    for l in dados.pols, t in janela
        # Entrada total vinda das fábricas <= MAX_DESCARREGAMENTO do POL
        @constraint(model,
            sum(e_fab[(f, p), t] for f in fabs_do_pol[l] for p in prods_da_fab[f]) <= dados.max_descarregamento[l] + folga_desc_pol[l, t]
        )
        # Saída total para navios <= MAX_CARREGAMENTO do POL
        @constraint(model,
            sum(Y_rota[(r_i, l, d), t] for (r_i, r) in enumerate(dados.rotas) if l in r.origens for d in r.destinos) <= dados.max_carregamento[l]
        )
        # Estoque total <= MAX_ESTOQUE do POL
        @constraint(model, sum(I_pol[l, p, t] for p in 1:n_prods) <= dados.max_estoque[l] + folga_est_pol[l, t])

        for p in 1:n_prods
            est_ant = (t == t_atual) ? get(estado.estoque_pol, (l, p), 0.0) : I_pol[l, p, t - 1]
            entrada_fab = sum(e_fab[(f, p), t] for f in fabs_do_pol[l] if p in prods_da_fab[f]; init=AffExpr(0.0))
            dest_taus = get(tuplas_f_por_lp, (l, p), Tuple{String,Int}[])
            saida_mar = sum(f_mar[(l, d, tau, p), t] for (d, tau) in dest_taus; init=AffExpr(0.0))
            @constraint(model, I_pol[l, p, t] == est_ant + entrada_fab - saida_mar)
        end
    end

    # --- 5.5 Balanço de Estoque nos PODs, Atendimento aos Clientes e Cobertura ---
    pares_dp = Tuple{String,Int}[]
    for p in 1:n_prods, d in pods_ativos_por_prod[p]
        push!(pares_dp, (d, p))
    end
    unique!(pares_dp)
    set_pares_dp = Set(pares_dp)

    tuplas_z = Tuple{String,String,Int,Int}[]
    pods_hab_kpt = Dict{Tuple{String,Int,Int},Vector{String}}()
    clientes_por_dp_t = Dict{Tuple{String,Int,Int},Vector{String}}()

    for t in janela
        for ((k, p, t_dem), dem_val) in dados.demanda
            if t_dem == t && dem_val > 0
                lista_d = String[]
                for d in dados.pods_do_cliente[k]
                    if (d, p) in set_pares_dp
                        push!(tuplas_z, (d, k, p, t))
                        push!(lista_d, d)
                        push!(get!(clientes_por_dp_t, (d, p, t), String[]), k)
                    end
                end
                pods_hab_kpt[(k, p, t)] = unique(lista_d)
            end
        end
    end
    unique!(tuplas_z)
    for k_dp in keys(clientes_por_dp_t)
        unique!(clientes_por_dp_t[k_dp])
    end

    @variable(model, z_ent[dkpt in tuplas_z] >= 0)
    @variable(model, I_pod[dp in pares_dp, t in janela] >= 0)
    @variable(model, u_cob[dp in pares_dp, t in janela] >= 0)

    # Limite de atendimento pela demanda de cada cliente
    for ((k, p, t), lista_d) in pods_hab_kpt
        dem_val = dados.demanda[(k, p, t)]
        if !isempty(lista_d)
            @constraint(model, sum(z_ent[(d, k, p, t)] for d in lista_d) <= dem_val)
        end
    end

    # Equações de balanço e cobertura nos PODs
    fator_cobertura = config.dias_cobertura_pod / 30.0
    for (d, p) in pares_dp, t in janela
        est_ant = (t == t_atual) ? get(estado.estoque_pod, (d, p), 0.0) : I_pod[(d, p), t - 1]

        chegada_fixa = get(estado.transito_pod, (d, p, t), 0.0)
        chegada_var = AffExpr(0.0)
        for tau in (1, 2)
            t_envio = t - tau
            if t_envio >= t_atual
                origs_l = get(tuplas_f_por_dp_tau, (d, p, tau), String[])
                for l in origs_l
                    add_to_expression!(chegada_var, 1.0, f_mar[(l, d, tau, p), t_envio])
                end
            end
        end

        clis_atend = get(clientes_por_dp_t, (d, p, t), String[])
        saida_cli = sum(z_ent[(d, k, p, t)] for k in clis_atend; init=AffExpr(0.0))

        @constraint(model, I_pod[(d, p), t] == est_ant + chegada_fixa + chegada_var - saida_cli)

        # Política de cobertura de estoque (15 dias de demanda escoada pelo POD)
        @constraint(model, I_pod[(d, p), t] + u_cob[(d, p), t] >= fator_cobertura * saida_cli)
    end

    # Capacidades operacionais agregadas nos PODs por mês (descarregamento, carregamento e armazenagem)
    prods_do_pod = Dict{String,Vector{Int}}()
    for (d, p) in pares_dp
        push!(get!(prods_do_pod, d, Int[]), p)
    end
    pods_com_prod = collect(keys(prods_do_pod))
    @variable(model, folga_pod[d in pods_com_prod, t in janela] >= 0)
    for (d, lista_p) in prods_do_pod, t in janela
        # Armazenagem máxima no POD
        @constraint(model, sum(I_pod[(d, p), t] for p in lista_p) <= dados.max_estoque[d] + folga_pod[d, t])
        # Carregamento máximo (expedição in-land para clientes) no POD
        @constraint(model,
            sum(z_ent[(d, k, p, t)] for p in lista_p for k in get(clientes_por_dp_t, (d, p, t), String[]); init=AffExpr(0.0)) <= dados.max_carregamento[d] + folga_pod[d, t]
        )
        # Descarregamento máximo de navios no POD no mês t
        desc_fixa = sum(get(estado.transito_pod, (d, p, t), 0.0) for p in lista_p)
        desc_var = AffExpr(0.0)
        for tau in (1, 2)
            t_envio = t - tau
            if t_envio >= t_atual
                for p in lista_p, l in get(tuplas_f_por_dp_tau, (d, p, tau), String[])
                    add_to_expression!(desc_var, 1.0, f_mar[(l, d, tau, p), t_envio])
                end
            end
        end
        @constraint(model, desc_fixa + desc_var <= dados.max_descarregamento[d] + folga_pod[d, t])
    end

    # --- 5.6 Função Objetivo (Maximizar Margem Total na Janela) ---
    receita_expr = sum(
        (dados.preco_venda[(k, p, t)] + config.penalidade_falta_demanda) * z_ent[(d, k, p, t)]
        for (d, k, p, t) in tuplas_z; init=AffExpr(0.0)
    )

    inland_expr = sum(
        dados.custo_inland[(k, d)] * z_ent[(d, k, p, t)]
        for (d, k, p, t) in tuplas_z; init=AffExpr(0.0)
    )

    frete_expr = AffExpr(0.0)
    for (r_i, r) in enumerate(dados.rotas), t in janela
        c = r.frete_base
        if config.usar_desconto_escalonado
            add_to_expression!(frete_expr, c,        w_faixa[r_i, 1, t])
            add_to_expression!(frete_expr, 0.90 * c, w_faixa[r_i, 2, t])
            add_to_expression!(frete_expr, 0.75 * c, w_faixa[r_i, 3, t])
        else
            add_to_expression!(frete_expr, c, w_faixa[r_i, 1, t] + w_faixa[r_i, 2, t] + w_faixa[r_i, 3, t])
        end
    end

    penal_cob_expr = sum(
        config.penalidade_cobertura * u_cob[(d, p), t]
        for (d, p) in pares_dp, t in janela; init=AffExpr(0.0)
    )

    custo_est_expr = config.custo_estocagem_mensal * (
        sum(I_fab[(f, p), t] for (f, p) in pares_fp, t in janela; init=AffExpr(0.0)) +
        sum(I_pol[l, p, t] for l in dados.pols, p in 1:n_prods, t in janela; init=AffExpr(0.0)) +
        sum(I_pod[(d, p), t] for (d, p) in pares_dp, t in janela; init=AffExpr(0.0))
    )

    # Penalidade alta para folgas operacionais no mês corrente e moderada no fim do lookahead
    penal_folga_expr = AffExpr(0.0)
    for t in janela
        peso = (t == t_atual) ? 25000.0 : 500.0
        for f in dados.fabricas
            add_to_expression!(penal_folga_expr, peso, folga_carreg_fab[f, t])
            add_to_expression!(penal_folga_expr, peso, folga_est_fab[f, t])
        end
        for l in dados.pols
            add_to_expression!(penal_folga_expr, peso, folga_est_pol[l, t])
            add_to_expression!(penal_folga_expr, peso, folga_desc_pol[l, t])
        end
        for d in pods_com_prod
            add_to_expression!(penal_folga_expr, peso, folga_pod[d, t])
        end
        for a in dados.armadores
            add_to_expression!(penal_folga_expr, 500000.0, folga_frota[a, t])
        end
    end

    @objective(model, Max, receita_expr - frete_expr - inland_expr - penal_cob_expr - custo_est_expr - penal_folga_expr)

    optimize!(model)

    # Contorno 1: se não achou solução inteira, tenta de novo com mais tempo,
    # mais esforço de heurísticas primais e gap mais folgado.
    if !has_values(model)
        println("   [mês $t_atual] sem solução no tempo inicial; retentando com mais tempo/heurísticas...")
        set_attribute(model, "time_limit", 4 * config.time_limit_por_mes)
        set_attribute(model, "mip_heuristic_effort", 0.5)
        set_attribute(model, "mip_rel_gap", 0.03)
        optimize!(model)
    end

    # Contorno 2: relaxa a integralidade dos navios do mês corrente, resolve o LP,
    # arredonda os navios para cima e resolve de novo com eles fixos.
    if !has_values(model)
        println("   [mês $t_atual] ainda sem solução inteira; usando relaxação + arredondamento de navios...")
        desfazer = relax_integrality(model)
        set_attribute(model, "time_limit", 4 * config.time_limit_por_mes)
        optimize!(model)
        if has_values(model)
            nav_fix = [ceil(Int, value(n_navios[r, t_atual]) - 1e-6) for r in R_idx]
            desfazer()
            for r in R_idx
                fix(n_navios[r, t_atual], nav_fix[r]; force=true)
            end
            optimize!(model)
        end
    end

    status = termination_status(model)
    if !has_values(model)
        error("Subproblema do mês $t_atual não encontrou solução viável (Status: $status).")
    end

    # --- 5.7 Extrair e Fixar as Melhores Decisões do Mês `t_atual` ---
    t = t_atual

    # 1. Produção e Estoques nas Fábricas no mês t
    dec_producao = NamedTuple[]
    for (f, p) in pares_fp
        q_prod = max(0.0, value(x_prod[(f, p), t]))
        q_env  = max(0.0, value(e_fab[(f, p), t]))
        q_est  = max(0.0, value(I_fab[(f, p), t]))
        estado.estoque_fab[(f, p)] = q_est
        if q_prod > 1e-3 || q_env > 1e-3 || q_est > 1e-3
            fam, lin = dados.produtos[p]
            push!(dec_producao, (
                mes = t,
                fabrica = f,
                pol = dados.pol_da_fabrica[f],
                familia = fam,
                linha = lin,
                producao_ton = round(q_prod, digits=2),
                envio_pol_ton = round(q_env, digits=2),
                estoque_final_fab_ton = round(q_est, digits=2)
            ))
        end
    end

    # 2. Estoques nos POLs no mês t
    dec_est_pol = NamedTuple[]
    for l in dados.pols, p in 1:n_prods
        q_est = max(0.0, value(I_pol[l, p, t]))
        estado.estoque_pol[(l, p)] = q_est
        if q_est > 1e-3
            fam, lin = dados.produtos[p]
            push!(dec_est_pol, (
                mes = t,
                pol = l,
                familia = fam,
                linha = lin,
                estoque_final_ton = round(q_est, digits=2)
            ))
        end
    end

    # 3. Embarques Marítimos, Desagregação Exata Rota-Produto e Atualização do Trânsito / Frota
    dec_embarques = NamedTuple[]
    custo_frete_mes = 0.0
    total_navios_mes = 0

    for (r_i, r) in enumerate(dados.rotas)
        nav = Int(round(value(n_navios[r_i, t])))
        if nav > 0
            total_navios_mes += nav
            w1 = max(0.0, value(w_faixa[r_i, 1, t]))
            w2 = max(0.0, value(w_faixa[r_i, 2, t]))
            w3 = max(0.0, value(w_faixa[r_i, 3, t]))
            carga_r = w1 + w2 + w3
            custo_r = config.usar_desconto_escalonado ?
                r.frete_base * (w1 + 0.90 * w2 + 0.75 * w3) :
                r.frete_base * carga_r
            custo_frete_mes += custo_r

            # Registrar ocupação da frota nos meses futuros (t .. t + tempo_ida_volta - 1)
            for m_ocup in t:(t + r.tempo_ida_volta - 1)
                estado.navios_ocupados[(r.armador, m_ocup)] = get(estado.navios_ocupados, (r.armador, m_ocup), 0) + nav
            end

            # Desagregar o volume de cada par (l, d) da rota r entre os produtos embarcados em (l, d, tau)
            tau = r.tempo_ida
            t_chegada = t + tau
            orig_str = join(r.origens, ";")
            dest_str = join(r.destinos, ";")

            for l in r.origens, d in r.destinos
                vol_rld = max(0.0, value(Y_rota[(r_i, l, d), t]))
                if vol_rld > 1e-3
                    prods_ldt = get(tuplas_f_por_ldt, (l, d, tau), Int[])
                    soma_f = sum(max(0.0, value(f_mar[(l, d, tau, p), t])) for p in prods_ldt; init=0.0)
                    if soma_f > 1e-6
                        for p in prods_ldt
                            val_fp = max(0.0, value(f_mar[(l, d, tau, p), t]))
                            if val_fp > 1e-3
                                vol_p = vol_rld * (val_fp / soma_f)
                                custo_p = custo_r * (vol_p / max(carga_r, 1e-6))
                                fam, lin = dados.produtos[p]
                                push!(dec_embarques, (
                                    mes_embarque = t,
                                    mes_chegada = t_chegada,
                                    rota_id = r.id,
                                    armador = r.armador,
                                    origens_rota = orig_str,
                                    destinos_rota = dest_str,
                                    navios_rota = nav,
                                    pol_carregamento = l,
                                    pod_descarregamento = d,
                                    familia = fam,
                                    linha = lin,
                                    volume_ton = round(vol_p, digits=2),
                                    custo_frete_usd = round(custo_p, digits=2)
                                ))
                            end
                        end
                    end
                end
            end
        end
    end

    # Registrar no estado os volumes em trânsito que chegarão aos PODs em t+1 e t+2
    for (l, d, tau, p) in tuplas_f
        val_fp = max(0.0, value(f_mar[(l, d, tau, p), t]))
        if val_fp > 1e-5
            t_cheg = t + tau
            chave_tr = (d, p, t_cheg)
            estado.transito_pod[chave_tr] = get(estado.transito_pod, chave_tr, 0.0) + val_fp
        end
    end

    # 4. Atendimento aos Clientes no mês t
    dec_atendimento = NamedTuple[]
    receita_mes = 0.0
    inland_mes = 0.0
    dem_total_mes = 0.0
    dem_atend_mes = 0.0
    saida_por_dp = Dict{Tuple{String,Int},Float64}()

    for ((k, p, t_dem), dem_val) in dados.demanda
        if t_dem == t
            dem_total_mes += dem_val
            lista_d = get(pods_hab_kpt, (k, p, t), String[])
            preco = dados.preco_venda[(k, p, t)]
            fam, lin = dados.produtos[p]
            atend_kpt = 0.0
            for d in lista_d
                vol_z = max(0.0, value(z_ent[(d, k, p, t)]))
                if vol_z > 1e-3
                    atend_kpt += vol_z
                    dem_atend_mes += vol_z
                    c_in = dados.custo_inland[(k, d)]
                    rec_z = preco * vol_z
                    inl_z = c_in * vol_z
                    receita_mes += rec_z
                    inland_mes += inl_z
                    saida_por_dp[(d, p)] = get(saida_por_dp, (d, p), 0.0) + vol_z
                    push!(dec_atendimento, (
                        mes = t,
                        cliente = k,
                        pod = d,
                        familia = fam,
                        linha = lin,
                        demanda_total_cliente_ton = round(dem_val, digits=2),
                        volume_atendido_ton = round(vol_z, digits=2),
                        preco_venda_usd_ton = round(preco, digits=2),
                        custo_inland_usd_ton = round(c_in, digits=2),
                        receita_usd = round(rec_z, digits=2),
                        custo_inland_total_usd = round(inl_z, digits=2)
                    ))
                end
            end
        end
    end

    # 5. Estoques e Cobertura nos PODs no mês t
    dec_est_pod = NamedTuple[]
    deficit_cob_mes = 0.0
    for (d, p) in pares_dp
        q_est = max(0.0, value(I_pod[(d, p), t]))
        q_def = max(0.0, value(u_cob[(d, p), t]))
        estado.estoque_pod[(d, p)] = q_est
        deficit_cob_mes += q_def
        meta_cob = fator_cobertura * get(saida_por_dp, (d, p), 0.0)
        if q_est > 1e-3 || meta_cob > 1e-3
            fam, lin = dados.produtos[p]
            push!(dec_est_pod, (
                mes = t,
                pod = d,
                familia = fam,
                linha = lin,
                estoque_final_ton = round(q_est, digits=2),
                meta_cobertura_15d_ton = round(meta_cob, digits=2),
                deficit_cobertura_ton = round(q_def, digits=2)
            ))
        end
    end

    margem_mes = receita_mes - custo_frete_mes - inland_mes

    return DecisoesMes(
        t,
        receita_mes,
        custo_frete_mes,
        inland_mes,
        margem_mes,
        dem_total_mes,
        dem_atend_mes,
        deficit_cob_mes,
        total_navios_mes,
        dec_producao,
        dec_embarques,
        dec_est_pol,
        dec_est_pod,
        dec_atendimento
    )
end

# ------------------------------------------------------------------------------
# 6. Loop Principal do Rolling Horizon (Mês a Mês) e Exportação de Resultados
# ------------------------------------------------------------------------------
function executar_rolling_horizon(config::ConfigModelo = ConfigModelo())
    dados = carregar_dados(config)
    estado = inicializar_estado(dados)
    historico_decisoes = DecisoesMes[]

    println("\n========================================================================================")
    println(" INICIANDO RESOLUÇÃO SEQUENCIAL MÊS A MÊS (ROLLING HORIZON: 1 A $(config.horizonte_planejamento))")
    println("========================================================================================")
    println(@sprintf("%-4s | %-12s | %-12s | %-8s | %-6s | %-14s | %-12s | %-14s",
        "Mês", "Demanda(t)", "Atendido(t)", "% Atend.", "Navios", "Receita(USD)", "Frete(USD)", "Margem(USD)"))
    println("----------------------------------------------------------------------------------------")

    tempo_total = @elapsed begin
        for t in 1:config.horizonte_planejamento
            dec_mes = resolver_mes!(t, dados, config, estado)
            push!(historico_decisoes, dec_mes)
            pct = dec_mes.demanda_total > 0 ? 100.0 * dec_mes.demanda_atendida / dec_mes.demanda_total : 100.0
            println(@sprintf(" %2d  | %12.1f | %12.1f | %7.2f%% | %6d | %14.2f | %12.2f | %14.2f",
                t, dec_mes.demanda_total, dec_mes.demanda_atendida, pct,
                dec_mes.total_navios, dec_mes.receita_bruta, dec_mes.custo_frete, dec_mes.margem_operacional))
        end
    end

    println("----------------------------------------------------------------------------------------")
    rec_tot   = sum(d.receita_bruta for d in historico_decisoes)
    frete_tot = sum(d.custo_frete for d in historico_decisoes)
    inl_tot   = sum(d.custo_inland for d in historico_decisoes)
    marg_tot  = sum(d.margem_operacional for d in historico_decisoes)
    dem_tot   = sum(d.demanda_total for d in historico_decisoes)
    atend_tot = sum(d.demanda_atendida for d in historico_decisoes)
    nav_tot   = sum(d.total_navios for d in historico_decisoes)
    pct_tot   = dem_tot > 0 ? 100.0 * atend_tot / dem_tot : 100.0

    println(@sprintf("TOT  | %12.1f | %12.1f | %7.2f%% | %6d | %14.2f | %12.2f | %14.2f",
        dem_tot, atend_tot, pct_tot, nav_tot, rec_tot, frete_tot, marg_tot))
    println("========================================================================================")
    println(@sprintf("Tempo total de execução dos %d meses: %.2f segundos (Custo In-land Total: %.2f USD).",
        config.horizonte_planejamento, tempo_total, inl_tot))

    salvar_relatorios(historico_decisoes, config)
    return historico_decisoes, dados, estado
end

function salvar_relatorios(historico::Vector{DecisoesMes}, config::ConfigModelo)
    mkpath(config.diretorio_saida)

    # 1. Produção nas Fábricas
    df_prod = DataFrame(vcat([d.producao for d in historico]...))
    CSV.write(joinpath(config.diretorio_saida, "1_producao_fabricas.csv"), df_prod)

    # 2. Embarques e Rotas Marítimas
    df_emb = DataFrame(vcat([d.embarques for d in historico]...))
    CSV.write(joinpath(config.diretorio_saida, "2_embarques_rotas.csv"), df_emb)

    # 3. Estoques nos POLs e PODs
    df_est_pol = DataFrame(vcat([d.estoques_pol for d in historico]...))
    CSV.write(joinpath(config.diretorio_saida, "3_estoques_pol.csv"), df_est_pol)

    df_est_pod = DataFrame(vcat([d.estoques_pod for d in historico]...))
    CSV.write(joinpath(config.diretorio_saida, "4_estoques_pod.csv"), df_est_pod)

    # 4. Atendimento aos Clientes
    df_atend = DataFrame(vcat([d.atendimento for d in historico]...))
    CSV.write(joinpath(config.diretorio_saida, "5_atendimento_clientes.csv"), df_atend)

    # 5. Resumo Mensal Executivo
    df_resumo = DataFrame([
        (
            mes = d.mes,
            demanda_total_ton = round(d.demanda_total, digits=2),
            demanda_atendida_ton = round(d.demanda_atendida, digits=2),
            atendimento_pct = round(d.demanda_total > 0 ? 100.0 * d.demanda_atendida / d.demanda_total : 100.0, digits=2),
            navios_despachados = d.total_navios,
            receita_bruta_usd = round(d.receita_bruta, digits=2),
            custo_frete_maritimo_usd = round(d.custo_frete, digits=2),
            custo_inland_usd = round(d.custo_inland, digits=2),
            margem_operacional_usd = round(d.margem_operacional, digits=2),
            deficit_cobertura_ton = round(d.deficit_cobertura_total, digits=2)
        )
        for d in historico
    ])
    CSV.write(joinpath(config.diretorio_saida, "6_resumo_mensal.csv"), df_resumo)

    # 6. Desafio 2.4.4: Análise de Rentabilidade por Cliente (Rateando frete médio por POD/Produto)
    if nrow(df_atend) > 0
        frete_medio_global = sum(d.custo_frete for d in historico) / max(sum(d.demanda_atendida for d in historico), 1.0)
        gdf = groupby(df_atend, :cliente)
        df_cli = combine(gdf,
            :volume_atendido_ton => sum => :volume_total_ton,
            :receita_usd => sum => :receita_total_usd,
            :custo_inland_total_usd => sum => :custo_inland_total_usd
        )
        df_cli.frete_maritimo_estimado_usd = round.(df_cli.volume_total_ton .* frete_medio_global, digits=2)
        df_cli.margem_liquida_estimada_usd = round.(df_cli.receita_total_usd .- df_cli.custo_inland_total_usd .- df_cli.frete_maritimo_estimado_usd, digits=2)
        df_cli.deficitario = df_cli.margem_liquida_estimada_usd .< 0.0
        sort!(df_cli, :margem_liquida_estimada_usd)
        CSV.write(joinpath(config.diretorio_saida, "7_rentabilidade_clientes.csv"), df_cli)
    end

    println("\nRelatórios detalhados salvos em: ", config.diretorio_saida)
end

if abspath(PROGRAM_FILE) == @__FILE__
    executar_rolling_horizon()
end
