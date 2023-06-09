<?php
	// print_r($_POST);
	$email = $_POST['email'];
	$pass = $_POST['pass'];
	$form1data = $_POST['form1data'];
	$form2data = $_POST['form2data'];
    $words = explode(",", $form1data)[0];
    $wordb = explode(",", $form1data)[1];
    $wordc = explode(",", $form1data)[2];
    $par = explode(",", $form2data)[0];


	$con = new mysqli("localhost","root","Alisson1911&rox","datab");
	if($con->connect_error){
		die("Failed to connect : ".$con->connect_error);
	}else{
		$stmt = $con->prepare("select * from people where email = ?");
		$stmt->bind_param("s",$email);
		$stmt->execute();
		$stmt_result = $stmt->get_result();
		if($stmt_result->num_rows > 0){
			$data = $stmt_result->fetch_assoc();
			if($data['pass'] === $pass){
				$query = "UPDATE PEOPLE SET words='$words',wordb='$wordb',wordc='$wordc', par='$par'  WHERE email='$data[email]'";

				mysqli_query($con,$query);
				header("Location: ../popup/index.html");
				// echo "<p>This is a variable $par</p>";
			}
			else{
				echo "<h2>Invalid Email or password</h2>";
			}
		}
		else{
			echo "<h2>Invalid Email or password</h2>";
		}
	}
?>